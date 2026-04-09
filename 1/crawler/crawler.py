"""
Main crawler orchestrator — wires together all engine components.

Crawler.run() implements the BFS crawl loop:

  while pages < max_pages and frontier not exhausted:
      pop URL from frontier
      fetch (async, up to max_workers concurrent)
      parse HTML → extract links
      push unseen links into frontier (Bloom-filtered)
      persist page + links to ContentStore
      log CrawlEvent to MetricsLogger

The relevance_fn hook lets experiment owners (Person 2, Person 3) inject
topic scoring without touching this file.  It receives a ParsedPage and
returns True/False; the default accepts everything.

The frontier is also pluggable: pass a PriorityFrontier (or any
URLFrontier subclass) to swap in custom URL ordering.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

from .bloom import BloomFilter
from .fetcher import Fetcher, FetchResult
from .frontier import URLFrontier, BFSFrontier
from .metrics import CrawlEvent, MetricsLogger
from .parser import ParsedPage, parse_page
from .store import ContentStore

logger = logging.getLogger(__name__)


class Crawler:
    """
    Single-node async BFS web crawler.

    Parameters
    ----------
    seed_urls        : Starting URLs for the crawl.
    frontier         : URLFrontier instance (default: BFSFrontier).
    max_pages        : Stop after successfully fetching this many pages.
    max_workers      : Maximum concurrent in-flight HTTP requests.
    user_agent       : HTTP User-Agent header.
    rate_limit_delay : Minimum delay (seconds) between requests per domain.
    db_path          : Path to the SQLite content store.
    metrics_path     : Path for the output CSV metrics log.
    bloom_capacity   : Expected unique URLs for Bloom filter sizing.
    bloom_error_rate : Target false-positive rate for Bloom filter.
    relevance_fn     : Optional callable (ParsedPage) -> bool.
                       Used to populate is_relevant in the metrics CSV.
    allowed_domains  : If provided, only crawl URLs on these domains.
    """

    def __init__(
        self,
        seed_urls: List[str],
        frontier: Optional[URLFrontier] = None,
        max_pages: int = 1_000,
        max_workers: int = 10,
        user_agent: str = "InternetSysCrawler/1.0",
        rate_limit_delay: float = 1.0,
        db_path: str = "crawl.db",
        metrics_path: str = "crawl_metrics.csv",
        bloom_capacity: int = 1_000_000,
        bloom_error_rate: float = 0.01,
        relevance_fn: Optional[Callable[[ParsedPage], bool]] = None,
        allowed_domains: Optional[Set[str]] = None,
    ) -> None:
        self.seed_urls = list(seed_urls)
        self.frontier = frontier or BFSFrontier()
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.user_agent = user_agent
        self.rate_limit_delay = rate_limit_delay
        self.db_path = db_path
        self.metrics_path = metrics_path
        self.bloom = BloomFilter(capacity=bloom_capacity, error_rate=bloom_error_rate)
        self.relevance_fn: Callable[[ParsedPage], bool] = relevance_fn or (lambda _: True)
        self.allowed_domains = allowed_domains

        # Running counters (read by external code during a crawl)
        self.pages_crawled: int = 0
        self.pages_relevant: int = 0

    # ------------------------------------------------------------------
    # Domain filter
    # ------------------------------------------------------------------

    def _domain_allowed(self, url: str) -> bool:
        if not self.allowed_domains:
            return True
        from urllib.parse import urlparse
        return urlparse(url).netloc in self.allowed_domains

    # ------------------------------------------------------------------
    # Process a single URL
    # ------------------------------------------------------------------

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

            depth = metadata.get("depth", 0)
            for link in parsed.links:
                if link.url not in self.bloom and self._domain_allowed(link.url):
                    self.bloom.add(link.url)
                    await self.frontier.push(
                        link.url,
                        priority=0.0,
                        metadata={"depth": depth + 1, "src": url},
                    )
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
            "[%d/%d] %s %s (%.0f ms, %d B)%s",
            self.pages_crawled, self.max_pages,
            result.status_code or "ERR", url,
            result.latency_ms, result.byte_size,
            " [relevant]" if is_relevant else "",
        )

    # ------------------------------------------------------------------
    # Main crawl loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Execute the crawl.  Returns when max_pages is reached or the
        frontier is fully drained (whichever comes first).
        """
        # Seed
        for url in self.seed_urls:
            if url not in self.bloom and self._domain_allowed(url):
                self.bloom.add(url)
                await self.frontier.push(url, priority=0.0, metadata={"depth": 0})

        with ContentStore(self.db_path) as store, \
             MetricsLogger(self.metrics_path) as metrics:

            async with Fetcher(
                user_agent=self.user_agent,
                rate_limit_delay=self.rate_limit_delay,
            ) as fetcher:
                await self._drain(fetcher, store, metrics)

        logger.info(
            "Crawl finished. pages=%d relevant=%d bloom=%s",
            self.pages_crawled, self.pages_relevant, self.bloom.stats(),
        )

    async def _drain(
        self,
        fetcher: Fetcher,
        store: ContentStore,
        metrics: MetricsLogger,
    ) -> None:
        """
        Core event loop: dispatch tasks from the frontier, bounded by
        max_workers concurrency and max_pages total.

        Termination conditions (any one suffices):
          A) pages_crawled >= max_pages
          B) frontier is empty AND no tasks are in flight
        """
        pending: Set[asyncio.Task] = set()

        while True:
            # --- Fill up to max_workers ---
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

            # --- Termination check ---
            if not pending:
                # Frontier is empty and nothing in flight → done
                break
            if self.pages_crawled >= self.max_pages:
                # Cancel anything still waiting
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                break

            # --- Wait for at least one task to complete ---
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            # Propagate any unexpected exceptions to the caller
            for t in done:
                exc = t.exception()
                if exc:
                    logger.error("Task raised: %s", exc, exc_info=exc)
