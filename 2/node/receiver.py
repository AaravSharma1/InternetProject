"""
Redis pub/sub receiver — listens on this node's channel for laterally
forwarded URLs and injects them into the local frontier.

Runs in a daemon thread because Part 1's crawl loop owns the asyncio
event loop.  Uses ``asyncio.run_coroutine_threadsafe`` to bridge the
thread boundary safely.
"""

import asyncio
import json
import logging
import threading
from typing import Optional

import redis

from config import NODE_CHANNEL_PREFIX

logger = logging.getLogger(__name__)


class URLReceiver:
    """
    Subscribes to ``crawl:node:<node_id>`` and pushes incoming URLs
    into the local frontier.
    """

    def __init__(
        self,
        redis_url: str,
        node_id: str,
        frontier,
        comm_tracker=None,
    ) -> None:
        # Each subscriber needs its own Redis connection (blocking reads)
        self._redis_url = redis_url
        self._node_id = node_id
        self._frontier = frontier
        self._comm_tracker = comm_tracker
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._channel = f"{NODE_CHANNEL_PREFIX}{node_id}"
        self.urls_received = 0

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Must be called after the asyncio event loop is running."""
        self._loop = loop

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen, daemon=True, name=f"receiver-{self._node_id}"
        )
        self._thread.start()
        logger.info("Receiver started on channel %s", self._channel)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Receiver stopped (%d URLs received)", self.urls_received)

    def _listen(self) -> None:
        r = redis.Redis.from_url(self._redis_url, decode_responses=True)
        pubsub = r.pubsub()
        pubsub.subscribe(self._channel)

        try:
            while not self._stop_event.is_set():
                msg = pubsub.get_message(ignore_subscribe_messages=True,
                                         timeout=1.0)
                if msg is None:
                    continue
                if msg["type"] != "message":
                    continue

                payload = msg["data"]
                if self._comm_tracker:
                    byte_count = len(payload) if isinstance(payload, str) else len(str(payload))
                    self._comm_tracker.record_received(byte_count)

                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("Malformed message on %s: %s",
                                   self._channel, payload[:100])
                    continue

                url = data["url"]
                priority = data.get("priority", 0.0)
                metadata = data.get("metadata", {})
                metadata["forwarded_from"] = data.get("forwarded_from",
                                                       "unknown")

                if self._loop is not None:
                    future = asyncio.run_coroutine_threadsafe(
                        self._frontier.push(url, priority, metadata),
                        self._loop,
                    )
                    future.result(timeout=5)

                self.urls_received += 1
                logger.debug("Received forwarded URL: %s (priority=%.3f)",
                             url, priority)
        finally:
            pubsub.unsubscribe()
            pubsub.close()
            r.close()
