import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=True)
class _Entry:
    # negate priority so heapq (min-heap) pops highest score first
    neg_priority: float
    counter: int
    url: str = field(compare=False)
    metadata: dict = field(compare=False)


class MockFrontier:
    """
    Implements Person 1's frontier interface for standalone testing.

    push(url, priority, metadata)  ->  None
    pop()                          ->  (url, priority, metadata) | None

    mode="priority" uses a max-heap ordered by score.
    mode="bfs" uses a plain FIFO queue.
    """

    def __init__(self, mode: str = "priority"):
        if mode not in ("bfs", "priority"):
            raise ValueError(f"mode must be 'bfs' or 'priority', got {mode!r}")
        self.mode = mode
        self._heap: list[_Entry] = []
        self._queue: deque = deque()
        self._counter = 0
        self._seen: set[str] = set()

    def push(self, url: str, priority: float, metadata: dict) -> None:
        if url in self._seen:
            return
        self._seen.add(url)

        if self.mode == "bfs":
            self._queue.append((url, priority, metadata))
        else:
            heapq.heappush(self._heap, _Entry(-priority, self._counter, url, metadata))
        self._counter += 1

    def pop(self) -> Optional[tuple[str, float, dict]]:
        if self.mode == "bfs":
            if not self._queue:
                return None
            return self._queue.popleft()
        else:
            if not self._heap:
                return None
            entry = heapq.heappop(self._heap)
            return (entry.url, -entry.neg_priority, entry.metadata)

    def peek_top_k(self, k: int = 5) -> list[tuple[str, float]]:
        if self.mode != "priority":
            raise ValueError("peek_top_k is only available in priority mode.")
        top = heapq.nsmallest(k, self._heap)
        return [(e.url, -e.neg_priority) for e in top]

    def __len__(self) -> int:
        return len(self._queue) if self.mode == "bfs" else len(self._heap)

    def is_empty(self) -> bool:
        return len(self) == 0

    def seen_count(self) -> int:
        return len(self._seen)
