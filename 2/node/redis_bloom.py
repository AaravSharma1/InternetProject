"""
Distributed Bloom filter backed by a Redis bitmap.

Drop-in replacement for Part 1's local BloomFilter — supports the same
``add`` / ``__contains__`` API so that ``if url not in self.bloom`` works
unchanged in the crawler.

Atomicity: ``add_if_absent`` uses a server-side Lua script so that the
check-and-set is a single atomic operation.  This prevents two nodes from
both deciding a URL is unseen and both enqueuing it.
"""

import hashlib
from typing import Any, Dict

import redis

from config import (
    BLOOM_BIT_SIZE,
    BLOOM_CAPACITY,
    BLOOM_ERROR_RATE,
    BLOOM_NUM_HASHES,
    BLOOM_REDIS_KEY,
)

# Lua script: check all k bits; if any is 0, set them all and return 1 (new).
# If all bits were already 1, return 0 (already present).
_LUA_ADD_IF_ABSENT = """
local key   = KEYS[1]
local count = tonumber(ARGV[1])
local new   = false
for i = 2, count + 1 do
    if redis.call('GETBIT', key, tonumber(ARGV[i])) == 0 then
        new = true
        break
    end
end
if new then
    for i = 2, count + 1 do
        redis.call('SETBIT', key, tonumber(ARGV[i]), 1)
    end
    return 1
end
return 0
"""


class RedisBloomFilter:
    """
    Redis-backed Bloom filter with identical hash logic to Part 1's
    ``BloomFilter`` (MD5-based, same bit_size / num_hashes).
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        capacity: int = BLOOM_CAPACITY,
        error_rate: float = BLOOM_ERROR_RATE,
        key: str = BLOOM_REDIS_KEY,
    ) -> None:
        self.r = redis_client
        self.key = key
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_size = BLOOM_BIT_SIZE
        self.num_hashes = BLOOM_NUM_HASHES
        self._count = 0  # local estimate (not globally accurate)

        # Register the Lua script once
        self._add_if_absent_sha = self.r.script_load(_LUA_ADD_IF_ABSENT)

    # ── Hash computation (matches Part 1 exactly) ──────────────────
    def _positions(self, item: str):
        raw = item.encode("utf-8")
        positions = []
        for i in range(self.num_hashes):
            digest = hashlib.md5(raw + i.to_bytes(4, "big")).hexdigest()
            positions.append(int(digest, 16) % self.bit_size)
        return positions

    # ── Public API ─────────────────────────────────────────────────
    def add(self, item: str) -> None:
        """Set all k bits for *item* in the Redis bitmap."""
        pipe = self.r.pipeline(transaction=False)
        for pos in self._positions(item):
            pipe.setbit(self.key, pos, 1)
        pipe.execute()
        self._count += 1

    def __contains__(self, item: str) -> bool:
        """Return True if *item* was probably added; False if definitely not."""
        pipe = self.r.pipeline(transaction=False)
        for pos in self._positions(item):
            pipe.getbit(self.key, pos)
        return all(pipe.execute())

    def add_if_absent(self, item: str) -> bool:
        """
        Atomic check-and-set.  Returns True if the item was newly added
        (at least one bit was 0), False if it was already present.
        """
        positions = self._positions(item)
        result = self.r.evalsha(
            self._add_if_absent_sha,
            1,               # number of KEYS
            self.key,        # KEYS[1]
            len(positions),  # ARGV[1]: count of positions
            *positions,      # ARGV[2..]: the positions
        )
        if result:
            self._count += 1
        return bool(result)

    # ── Stats (for demo / debugging) ───────────────────────────────
    def stats(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "error_rate": self.error_rate,
            "bit_size": self.bit_size,
            "num_hashes": self.num_hashes,
            "items_added_local": self._count,
        }
