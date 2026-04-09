"""
Hash-based URL router with Redis pub/sub forwarding.

Determines which node owns a URL (via partition hash) and, when the URL
belongs to a remote node, serialises it and publishes to that node's
Redis channel.
"""

import json
import logging
from typing import Any, Dict, List, Set

import redis

from config import NODE_CHANNEL_PREFIX, TOTAL_PARTITIONS
from coordinator.hash_ring import url_to_partition

logger = logging.getLogger(__name__)


class URLRouter:
    """
    Routes discovered URLs to the correct node's frontier.

    Each URL is hashed to a partition (0–255).  If the partition belongs
    to the local node, the URL is handled locally.  Otherwise it is
    serialised as JSON and published to ``crawl:node:<owner_id>``.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        node_id: str,
        local_partitions: List[int],
        node_hosts: Dict[str, Dict],
        comm_tracker=None,
    ) -> None:
        self.r = redis_client
        self.node_id = node_id
        self._local_parts: Set[int] = set(local_partitions)
        self._node_hosts = node_hosts       # {node_id: {host, port, partitions}}
        self._part_to_node: Dict[int, str] = {}
        self._comm_tracker = comm_tracker
        self._rebuild_partition_map()

    def _rebuild_partition_map(self) -> None:
        self._part_to_node.clear()
        for nid, info in self._node_hosts.items():
            for p in info.get("partitions", []):
                self._part_to_node[p] = nid

    def update_assignment(
        self,
        local_partitions: List[int],
        node_hosts: Dict[str, Dict],
    ) -> None:
        """Called when the coordinator publishes a new partition mapping."""
        self._local_parts = set(local_partitions)
        self._node_hosts = node_hosts
        self._rebuild_partition_map()
        logger.info("Assignment updated: %d local partitions", len(self._local_parts))

    def get_owner(self, url: str) -> str:
        """Return the node_id that should crawl *url*."""
        part = url_to_partition(url)
        return self._part_to_node.get(part, self.node_id)

    def is_local(self, url: str) -> bool:
        """True if *url* hashes to a partition owned by this node."""
        return url_to_partition(url) in self._local_parts

    def forward_url(
        self,
        url: str,
        priority: float,
        metadata: Dict[str, Any],
    ) -> None:
        """Publish the URL to the owning node's Redis channel."""
        owner = self.get_owner(url)
        if owner == self.node_id:
            return  # shouldn't happen, but guard anyway

        channel = f"{NODE_CHANNEL_PREFIX}{owner}"
        payload = json.dumps({
            "url": url,
            "priority": priority,
            "metadata": metadata,
            "forwarded_from": self.node_id,
        })
        self.r.publish(channel, payload)

        if self._comm_tracker:
            self._comm_tracker.record_sent(len(payload))

        logger.debug("Forwarded %s → %s (%d bytes)", url, owner, len(payload))
