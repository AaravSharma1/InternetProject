"""
Cross-node communication overhead tracker.

Records bytes sent and received per node per minute, using Redis counters
so that the experiment runner can aggregate across all nodes.
"""

import time
from typing import Dict, List

import redis

from config import COMM_STATS_PREFIX


def _minute_bucket() -> int:
    """Return the current minute as a Unix timestamp (floored to 60s)."""
    return int(time.time() / 60) * 60


class CommTracker:
    """
    Lightweight wrapper around Redis INCRBY for communication accounting.

    Keys: ``crawl:comm:<node_id>:<minute_ts>:sent``
          ``crawl:comm:<node_id>:<minute_ts>:recv``
    """

    def __init__(self, redis_client: redis.Redis, node_id: str) -> None:
        self.r = redis_client
        self.node_id = node_id
        self._local_sent = 0
        self._local_recv = 0

    def record_sent(self, byte_count: int) -> None:
        bucket = _minute_bucket()
        key = f"{COMM_STATS_PREFIX}{self.node_id}:{bucket}:sent"
        self.r.incrby(key, byte_count)
        self.r.expire(key, 3600)  # auto-expire after 1 hour
        self._local_sent += byte_count

    def record_received(self, byte_count: int) -> None:
        bucket = _minute_bucket()
        key = f"{COMM_STATS_PREFIX}{self.node_id}:{bucket}:recv"
        self.r.incrby(key, byte_count)
        self.r.expire(key, 3600)
        self._local_recv += byte_count

    @property
    def total_sent(self) -> int:
        return self._local_sent

    @property
    def total_received(self) -> int:
        return self._local_recv

    def get_stats(self) -> Dict[str, int]:
        """Return cumulative local totals."""
        return {
            "node_id": self.node_id,
            "total_sent_bytes": self._local_sent,
            "total_received_bytes": self._local_recv,
        }

    def get_minute_stats(self) -> List[Dict]:
        """Scan Redis for all minute buckets belonging to this node."""
        pattern = f"{COMM_STATS_PREFIX}{self.node_id}:*:sent"
        results = []
        for key in self.r.scan_iter(match=pattern):
            key_str = key.decode() if isinstance(key, bytes) else key
            parts = key_str.split(":")
            minute_ts = int(parts[3])
            sent_key = key_str
            recv_key = key_str.replace(":sent", ":recv")
            sent = int(self.r.get(sent_key) or 0)
            recv = int(self.r.get(recv_key) or 0)
            results.append({
                "minute": minute_ts,
                "sent_bytes": sent,
                "recv_bytes": recv,
            })
        return sorted(results, key=lambda d: d["minute"])
