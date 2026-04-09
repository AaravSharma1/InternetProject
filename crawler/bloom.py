"""
Bloom filter for URL deduplication.

Pure-Python implementation — no external deps beyond hashlib.

Usage:
    bf = BloomFilter(capacity=1_000_000, error_rate=0.01)
    bf.add("https://example.com/page")
    "https://example.com/page" in bf   # True  (definite)
    "https://never-seen.com"  in bf   # False (with high probability)
    print(bf.stats())
"""

import hashlib
import math
from typing import Dict, Any


class BloomFilter:
    """
    Space-efficient probabilistic set membership test.

    - No false negatives: if item was added, `in` always returns True.
    - Possible false positives: `in` may return True for items never added.
    - Estimated FPR is tracked and exposed via stats().

    Hash strategy: k independent positions are derived from a single MD5
    by hashing `item || i` for i in 0..k-1.  This avoids k full hash
    computations while keeping independence adequate for a Bloom filter.
    """

    def __init__(self, capacity: int = 1_000_000,
                 error_rate: float = 0.01) -> None:
        if not (0 < error_rate < 1):
            raise ValueError("error_rate must be between 0 and 1 exclusive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.target_error_rate = error_rate

        # Optimal parameters
        # m = -n * ln(p) / (ln 2)^2
        self.bit_size: int = max(1,
            int(-capacity * math.log(error_rate) / (math.log(2) ** 2)))
        # k = (m/n) * ln 2
        self.num_hashes: int = max(1,
            int((self.bit_size / capacity) * math.log(2)))

        self._bytes = bytearray(math.ceil(self.bit_size / 8))
        self._count: int = 0           # items added (for FPR estimate)
        self._bits_set: int = 0        # popcount (for fill ratio)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _positions(self, item: str):
        """Yield k bit positions for *item*."""
        raw = item.encode("utf-8")
        for i in range(self.num_hashes):
            digest = hashlib.md5(raw + i.to_bytes(4, "big")).hexdigest()
            yield int(digest, 16) % self.bit_size

    def _get(self, pos: int) -> bool:
        return bool(self._bytes[pos >> 3] & (1 << (pos & 7)))

    def _set(self, pos: int) -> None:
        byte_idx = pos >> 3
        mask = 1 << (pos & 7)
        if not (self._bytes[byte_idx] & mask):
            self._bytes[byte_idx] |= mask
            self._bits_set += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, item: str) -> None:
        """Add *item* to the filter.  Always call this before checking `in`."""
        for pos in self._positions(item):
            self._set(pos)
        self._count += 1

    def __contains__(self, item: str) -> bool:
        """Return True if *item* was probably added; False if definitely not."""
        return all(self._get(pos) for pos in self._positions(item))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def estimated_fpr(self) -> float:
        """
        Current false-positive rate estimate based on actual fill:
            fpr = (1 - e^(-k * n / m))^k
        where n = items added, m = bit_size, k = num_hashes.
        """
        if self.bit_size == 0 or self._count == 0:
            return 0.0
        exponent = -self.num_hashes * self._count / self.bit_size
        return (1.0 - math.exp(exponent)) ** self.num_hashes

    def stats(self) -> Dict[str, Any]:
        fill = self._bits_set / self.bit_size if self.bit_size else 0.0
        return {
            "capacity": self.capacity,
            "target_error_rate": self.target_error_rate,
            "bit_size": self.bit_size,
            "num_hashes": self.num_hashes,
            "items_added": self._count,
            "bits_set": self._bits_set,
            "fill_ratio": round(fill, 6),
            "estimated_fpr": round(self.estimated_fpr, 8),
        }
