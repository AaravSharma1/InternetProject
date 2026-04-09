"""
Link-Priority Frontier — URLs ranked by inbound link count.

Implements Part 1's URLFrontier interface via a min-heap with *negated*
counts so that the URL with the highest inbound count is popped first.

Uses the **lazy deletion** pattern: every time a URL's count increments a
fresh heap entry is pushed.  On ``pop``, stale entries (whose stored count
no longer matches the current count) are silently discarded.
"""

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "1"))
from crawler.frontier import URLFrontier


@dataclass(order=True)
class _HeapItem:
    neg_count: int                                    # negated inbound count
    seq: int                                          # tie-break by insertion order
    url: str = field(compare=False)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)


class LinkPriorityFrontier(URLFrontier):
    """
    Priority frontier where URLs with more inbound links are fetched first.

    The ``priority`` parameter in ``push()`` is *ignored* — priority is
    derived from ``_inbound_counts`` so callers don't need special logic.
    Call ``increment_count(url)`` explicitly when a URL is discovered from
    a new source page, then ``push()`` it.
    """

    def __init__(self) -> None:
        self._heap: list = []
        self._seq: int = 0
        self._lock = asyncio.Lock()
        self._has_items = asyncio.Event()

        self._inbound_counts: Dict[str, int] = {}
        self._in_frontier: Set[str] = set()  # URLs currently in the heap

    # ── Count management ───────────────────────────────────────────
    def increment_count(self, url: str) -> int:
        """Increment and return the new inbound link count for *url*."""
        self._inbound_counts[url] = self._inbound_counts.get(url, 0) + 1
        return self._inbound_counts[url]

    def get_count(self, url: str) -> int:
        return self._inbound_counts.get(url, 0)

    # ── URLFrontier interface ──────────────────────────────────────
    async def push(self, url: str, priority: float = 0.0,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        async with self._lock:
            count = self._inbound_counts.get(url, 1)
            heapq.heappush(
                self._heap,
                _HeapItem(neg_count=-count, seq=self._seq,
                          url=url, metadata=metadata or {}),
            )
            self._seq += 1
            self._in_frontier.add(url)
            self._has_items.set()

    async def pop(self) -> Tuple[str, Dict[str, Any]]:
        while True:
            async with self._lock:
                while self._heap:
                    item = heapq.heappop(self._heap)
                    current_count = self._inbound_counts.get(item.url, 1)
                    # Accept if this entry reflects the current count
                    if -item.neg_count == current_count:
                        self._in_frontier.discard(item.url)
                        if not self._heap:
                            self._has_items.clear()
                        return item.url, item.metadata
                    # Stale entry — discard and continue
                self._has_items.clear()
            await self._has_items.wait()

    def __len__(self) -> int:
        return len(self._in_frontier)

    def empty(self) -> bool:
        return len(self._in_frontier) == 0
