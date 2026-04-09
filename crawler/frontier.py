"""
URL Frontier — pluggable priority queue for the crawl scheduler.

Interface:
    push(url, priority, metadata)  — add a URL to the frontier
    pop()                          — remove and return (url, metadata)
    __len__() / empty()            — size helpers

BFSFrontier  : plain FIFO (asyncio.Queue) — the BFS baseline.
PriorityFrontier : min-heap; lower priority value = dequeued first.
                   Person 2 (link-count scoring) and Person 3 (semantic
                   scoring) swap this in without touching Crawler code.
"""

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


class URLFrontier:
    """Abstract base — defines the interface every frontier must implement."""

    async def push(self, url: str, priority: float = 0.0,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    async def pop(self) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def empty(self) -> bool:
        return len(self) == 0


# ---------------------------------------------------------------------------
# BFS (FIFO) Frontier
# ---------------------------------------------------------------------------

class BFSFrontier(URLFrontier):
    """
    First-In First-Out frontier — implements BFS crawl order.

    Wraps asyncio.Queue so pop() naturally awaits when empty, which
    lets the crawler loop sleep without busy-waiting.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()

    async def push(self, url: str, priority: float = 0.0,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        await self._queue.put((url, metadata or {}))

    async def pop(self) -> Tuple[str, Dict[str, Any]]:
        return await self._queue.get()

    def task_done(self) -> None:
        """Signal that a previously pop()-ed item has been fully processed."""
        self._queue.task_done()

    async def join(self) -> None:
        """Block until all items have been processed (task_done called for each)."""
        await self._queue.join()

    def __len__(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()


# ---------------------------------------------------------------------------
# Priority Frontier
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _HeapItem:
    priority: float
    seq: int                                      # tie-break by insertion order
    url: str = field(compare=False)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)


class PriorityFrontier(URLFrontier):
    """
    Min-heap priority queue.  Lower priority value → popped first.

    Thread-safe via asyncio.Lock; pop() suspends until an item arrives.
    """

    def __init__(self) -> None:
        self._heap: list = []
        self._seq: int = 0
        self._lock = asyncio.Lock()
        self._has_items = asyncio.Event()

    async def push(self, url: str, priority: float = 0.0,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        async with self._lock:
            heapq.heappush(
                self._heap,
                _HeapItem(priority=priority, seq=self._seq,
                          url=url, metadata=metadata or {})
            )
            self._seq += 1
            self._has_items.set()

    async def pop(self) -> Tuple[str, Dict[str, Any]]:
        while True:
            async with self._lock:
                if self._heap:
                    item = heapq.heappop(self._heap)
                    if not self._heap:
                        self._has_items.clear()
                    return item.url, item.metadata
            await self._has_items.wait()

    def __len__(self) -> int:
        return len(self._heap)

    def empty(self) -> bool:
        return len(self._heap) == 0
