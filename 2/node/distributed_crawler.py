"""
Distributed crawler node — wraps Part 1's single-node Crawler.

Subclasses ``Crawler`` and overrides:
  * ``_process()`` — adds hash-based URL routing (local vs. forward)
  * ``run()``      — wraps with coordinator registration, receiver, heartbeat

Supports three crawl modes:
  * **bfs**            — distributed BFS (baseline)
  * **link_priority**  — URLs ranked by inbound link count
  * **semantic**       — URLs ranked by embedding similarity (Part 3)
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

import redis
import requests

# ── Path setup so we can import Part 1 and Part 3 ──────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "1"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "3"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "2"))

from crawler.crawler import Crawler
from crawler.fetcher import Fetcher, FetchResult
from crawler.frontier import BFSFrontier, PriorityFrontier, URLFrontier
from crawler.metrics import CrawlEvent, MetricsLogger
from crawler.parser import ParsedPage, parse_page
from crawler.store import ContentStore

from config import (
    ASSIGNMENT_CHANNEL,
    HEARTBEAT_INTERVAL,
    NODE_CHANNEL_PREFIX,
    REDIS_URL,
)
from node.comm_tracker import CommTracker
from node.link_priority import LinkPriorityFrontier
from node.receiver import URLReceiver
from node.redis_bloom import RedisBloomFilter
from node.url_router import URLRouter

logger = logging.getLogger(__name__)


class DistributedCrawler(Crawler):
    """
    Multi-node distributed crawler.

    Extends Part 1's ``Crawler`` with:
      - Redis-based Bloom filter (shared across nodes)
      - Hash-partitioned URL routing with lateral forwarding
      - Link-priority and semantic scoring modes
      - Coordinator registration and heartbeat
    """

    def __init__(
        self,
        seed_urls: List[str],
        node_id: str,
        coordinator_url: str,
        redis_url: str = REDIS_URL,
        mode: str = "bfs",
        topic: Optional[str] = None,
        max_pages: int = 1_000,
        max_workers: int = 10,
        user_agent: str = "InternetSysCrawler/1.0",
        rate_limit_delay: float = 1.0,
        db_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        bloom_capacity: int = 1_000_000,
        bloom_error_rate: float = 0.01,
        relevance_fn: Optional[Callable[[ParsedPage], bool]] = None,
        allowed_domains: Optional[Set[str]] = None,
        node_port: int = 6001,
    ) -> None:
        # Choose frontier based on mode
        if mode == "link_priority":
            frontier: URLFrontier = LinkPriorityFrontier()
        elif mode == "semantic":
            frontier = PriorityFrontier()
        else:
            frontier = BFSFrontier()

        # Default per-node file paths
        if db_path is None:
            db_path = f"{node_id}.db"
        if metrics_path is None:
            metrics_path = f"{node_id}_metrics.csv"

        super().__init__(
            seed_urls=seed_urls,
            frontier=frontier,
            max_pages=max_pages,
            max_workers=max_workers,
            user_agent=user_agent,
            rate_limit_delay=rate_limit_delay,
            db_path=db_path,
            metrics_path=metrics_path,
            bloom_capacity=bloom_capacity,
            bloom_error_rate=bloom_error_rate,
            relevance_fn=relevance_fn,
            allowed_domains=allowed_domains,
        )

        self.node_id = node_id
        self.node_port = node_port
        self.coordinator_url = coordinator_url.rstrip("/")
        self.redis_url = redis_url
        self.mode = mode
        self.topic = topic

        # Redis client
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)

        # Replace local Bloom with distributed Redis Bloom
        self.bloom = RedisBloomFilter(self._redis, capacity=bloom_capacity,
                                       error_rate=bloom_error_rate)

        # Communication tracker
        self.comm_tracker = CommTracker(self._redis, node_id)

        # These are set during registration
        self._local_partitions: List[int] = []
        self._all_nodes: Dict[str, Dict] = {}
        self.url_router: Optional[URLRouter] = None
        self.receiver: Optional[URLReceiver] = None

        # Semantic mode (lazy-loaded)
        self._semantic_scorer = None
        self._topic_centroid = None

        # Heartbeat control
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Assignment listener control
        self._assignment_stop = threading.Event()
        self._assignment_thread: Optional[threading.Thread] = None

    # ── Coordinator interaction ─────────────────────────────────────

    def _register(self) -> None:
        resp = requests.post(f"{self.coordinator_url}/register", json={
            "node_id": self.node_id,
            "host": "localhost",
            "port": self.node_port,
        })
        resp.raise_for_status()
        data = resp.json()
        self._local_partitions = data["partitions"]
        self._all_nodes = data["all_nodes"]

        # Build per-node info dict for the router
        node_hosts = {}
        for nid, parts in self._all_nodes.items():
            node_hosts[nid] = {"partitions": parts}

        self.url_router = URLRouter(
            self._redis, self.node_id,
            self._local_partitions, node_hosts,
            comm_tracker=self.comm_tracker,
        )
        logger.info("Registered %s — %d partitions assigned",
                     self.node_id, len(self._local_partitions))

    def _deregister(self) -> None:
        try:
            requests.post(f"{self.coordinator_url}/deregister",
                          json={"node_id": self.node_id}, timeout=5)
        except Exception as e:
            logger.warning("Deregister failed: %s", e)

    def _start_heartbeat(self) -> None:
        self._heartbeat_stop.clear()

        def _beat():
            while not self._heartbeat_stop.is_set():
                try:
                    requests.post(f"{self.coordinator_url}/heartbeat", json={
                        "node_id": self.node_id,
                        "pages_crawled": self.pages_crawled,
                        "frontier_size": len(self.frontier),
                    }, timeout=5)
                except Exception as e:
                    logger.warning("Heartbeat failed: %s", e)
                self._heartbeat_stop.wait(HEARTBEAT_INTERVAL)

        self._heartbeat_thread = threading.Thread(target=_beat, daemon=True,
                                                   name=f"heartbeat-{self.node_id}")
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)

    def _start_assignment_listener(self) -> None:
        """Listen for rebalance events from the coordinator."""
        self._assignment_stop.clear()

        def _listen():
            r = redis.Redis.from_url(self.redis_url, decode_responses=True)
            ps = r.pubsub()
            ps.subscribe(ASSIGNMENT_CHANNEL)
            try:
                while not self._assignment_stop.is_set():
                    msg = ps.get_message(ignore_subscribe_messages=True,
                                         timeout=1.0)
                    if msg and msg["type"] == "message":
                        try:
                            data = json.loads(msg["data"])
                            nodes = data.get("nodes", {})
                            my_info = nodes.get(self.node_id)
                            if my_info:
                                self._local_partitions = my_info["partitions"]
                                node_hosts = {
                                    nid: {"partitions": info["partitions"]}
                                    for nid, info in nodes.items()
                                }
                                if self.url_router:
                                    self.url_router.update_assignment(
                                        self._local_partitions, node_hosts)
                                logger.info("Rebalanced: now %d partitions",
                                            len(self._local_partitions))
                        except Exception as e:
                            logger.warning("Bad assignment update: %s", e)
            finally:
                ps.unsubscribe()
                ps.close()
                r.close()

        self._assignment_thread = threading.Thread(
            target=_listen, daemon=True, name=f"assign-{self.node_id}")
        self._assignment_thread.start()

    def _stop_assignment_listener(self) -> None:
        self._assignment_stop.set()
        if self._assignment_thread:
            self._assignment_thread.join(timeout=5)

    # ── Semantic mode initialisation ────────────────────────────────

    def _init_semantic(self) -> None:
        if self.mode != "semantic":
            return
        try:
            from semantic_prioritizer import SemanticPrioritizer
            self._semantic_scorer = SemanticPrioritizer()
            seed_desc = [self.topic] if self.topic else ["web technology"]
            self._topic_centroid = self._semantic_scorer.init_centroid(seed_desc)
            logger.info("Semantic mode: centroid initialised from '%s'",
                        self.topic)
        except ImportError:
            logger.error("Could not import SemanticPrioritizer from Part 3 — "
                         "falling back to BFS priorities")
            self.mode = "bfs"

    # ── Priority computation ────────────────────────────────────────

    def _compute_priority(self, link, src_url: str) -> float:
        """Return a priority value for the frontier (lower = popped first)."""
        if self.mode == "link_priority":
            if isinstance(self.frontier, LinkPriorityFrontier):
                count = self.frontier.increment_count(link.url)
                return float(-count)
            return 0.0

        if self.mode == "semantic" and self._semantic_scorer and self._topic_centroid is not None:
            context = self._semantic_scorer.build_url_context(
                link.anchor_text, link.context, link.url)
            score = self._semantic_scorer.score(context, self._topic_centroid)
            return -score  # negate: higher similarity → lower priority value → popped first

        return 0.0  # BFS

    # ── Override _process to add distributed routing ────────────────

    async def _process(
        self,
        url: str,
        metadata: Dict[str, Any],
        fetcher: Fetcher,
        store: ContentStore,
        metrics: MetricsLogger,
    ) -> None:
        logger.debug("Fetching %s (depth=%s)", url, metadata.get("depth", "?"))
        result: FetchResult = await fetcher.fetch(url)

        is_relevant = False
        parsed: Optional[ParsedPage] = None

        success = (
            result.html is not None
            and result.status_code is not None
            and 200 <= result.status_code < 300
        )

        if success:
            parsed = parse_page(result.html, url)
            is_relevant = self.relevance_fn(parsed)

            store.save_page(
                url=url,
                fetch_time=result.fetch_time,
                status_code=result.status_code,
                byte_size=result.byte_size,
                title=parsed.title,
                html=result.html,
                text=parsed.text,
            )
            store.save_links(url, parsed.links)

            # Update semantic centroid if relevant
            if (is_relevant and self.mode == "semantic"
                    and self._semantic_scorer and self._topic_centroid is not None):
                self._topic_centroid = self._semantic_scorer.update_centroid(
                    self._topic_centroid, parsed.text)

            # ── Distributed link routing (the key override) ────────
            depth = metadata.get("depth", 0)
            for link in parsed.links:
                if not self._domain_allowed(link.url):
                    continue
                # Atomic check-and-set in Redis Bloom
                if not self.bloom.add_if_absent(link.url):
                    continue  # already seen globally

                priority = self._compute_priority(link, url)
                link_meta = {"depth": depth + 1, "src": url}

                if self.url_router and not self.url_router.is_local(link.url):
                    # Forward to the owning node
                    self.url_router.forward_url(link.url, priority, link_meta)
                else:
                    # Local — push into our frontier
                    await self.frontier.push(link.url, priority, link_meta)
        else:
            store.save_page(
                url=url,
                fetch_time=result.fetch_time,
                status_code=result.status_code,
                byte_size=result.byte_size,
                error=result.error or "fetch failed",
            )

        self.pages_crawled += 1
        if is_relevant:
            self.pages_relevant += 1

        event = CrawlEvent(
            timestamp=result.fetch_time,
            url=url,
            bytes_downloaded=result.byte_size,
            fetch_latency_ms=result.latency_ms,
            status_code=result.status_code or 0,
            is_relevant=is_relevant,
            cumulative_pages=self.pages_crawled,
            cumulative_relevant=self.pages_relevant,
            error=result.error or "",
        )
        metrics.log(event)
        logger.info(
            "[%s %d/%d] %s %s (%.0f ms, %d B)%s",
            self.node_id,
            self.pages_crawled, self.max_pages,
            result.status_code or "ERR", url,
            result.latency_ms, result.byte_size,
            " [relevant]" if is_relevant else "",
        )

    # ── Override _drain to handle distributed termination ───────────

    async def _drain(
        self,
        fetcher: Fetcher,
        store: ContentStore,
        metrics: MetricsLogger,
    ) -> None:
        """
        Extended drain loop with a grace period for URLs in transit.

        In distributed mode the frontier may momentarily be empty while
        forwarded URLs are travelling through Redis pub/sub.  We wait
        a short time before declaring the crawl finished.
        """
        pending: Set[asyncio.Task] = set()
        empty_since: Optional[float] = None
        GRACE_PERIOD = 5.0  # seconds to wait for forwarded URLs

        while True:
            # Fill up to max_workers
            while (
                len(pending) < self.max_workers
                and not self.frontier.empty()
                and self.pages_crawled + len(pending) < self.max_pages
            ):
                url, metadata = await self.frontier.pop()
                task = asyncio.create_task(
                    self._process(url, metadata, fetcher, store, metrics),
                    name=f"crawl:{url}",
                )
                pending.add(task)
                empty_since = None  # reset grace timer

            # Termination check
            if not pending and self.frontier.empty():
                if empty_since is None:
                    empty_since = time.time()
                elif time.time() - empty_since > GRACE_PERIOD:
                    break
                # Brief sleep to allow forwarded URLs to arrive
                await asyncio.sleep(0.5)
                continue
            elif not pending:
                await asyncio.sleep(0.1)
                continue

            if self.pages_crawled >= self.max_pages:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                break

            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                exc = t.exception()
                if exc:
                    logger.error("Task raised: %s", exc, exc_info=exc)

    # ── Override run() to add distributed setup/teardown ────────────

    async def run(self) -> None:
        # Register with coordinator
        self._register()

        # Init semantic scorer if needed
        self._init_semantic()

        # Start receiver (pub/sub listener)
        self.receiver = URLReceiver(
            self.redis_url, self.node_id, self.frontier,
            comm_tracker=self.comm_tracker,
        )
        loop = asyncio.get_running_loop()
        self.receiver.set_loop(loop)
        self.receiver.start()

        # Start heartbeat
        self._start_heartbeat()

        # Start assignment rebalance listener
        self._start_assignment_listener()

        try:
            # Seed frontier — only URLs that hash to our partitions
            for url in self.seed_urls:
                if self._domain_allowed(url) and self.bloom.add_if_absent(url):
                    if self.url_router and not self.url_router.is_local(url):
                        self.url_router.forward_url(
                            url, 0.0, {"depth": 0, "src": "seed"})
                    else:
                        await self.frontier.push(
                            url, priority=0.0, metadata={"depth": 0})

            with ContentStore(self.db_path) as store, \
                 MetricsLogger(self.metrics_path) as metrics:
                async with Fetcher(
                    user_agent=self.user_agent,
                    rate_limit_delay=self.rate_limit_delay,
                ) as fetcher:
                    await self._drain(fetcher, store, metrics)

        finally:
            self._stop_heartbeat()
            self._stop_assignment_listener()
            if self.receiver:
                self.receiver.stop()
            self._deregister()

            logger.info(
                "Node %s finished. pages=%d relevant=%d "
                "forwarded_recv=%d comm_sent=%d comm_recv=%d",
                self.node_id, self.pages_crawled, self.pages_relevant,
                self.receiver.urls_received if self.receiver else 0,
                self.comm_tracker.total_sent,
                self.comm_tracker.total_received,
            )
