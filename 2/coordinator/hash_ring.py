"""
Hash-ring partition manager.

Distributes a fixed set of 256 virtual partitions evenly across registered
nodes.  URL ownership is determined by ``hash(url) % 256``.

When a node joins or leaves, only partition assignments change — the URL
hash function stays the same, so the distributed Bloom filter remains valid.
"""

import hashlib
from typing import Dict, List, Optional

from config import TOTAL_PARTITIONS


def url_to_partition(url: str) -> int:
    """Deterministic partition id for *url*.  Must match across all nodes."""
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    return int(digest, 16) % TOTAL_PARTITIONS


class HashRing:
    """
    Manages the mapping from virtual partitions → node IDs.

    All methods are synchronous and meant to run inside the coordinator.
    """

    def __init__(self, total_partitions: int = TOTAL_PARTITIONS) -> None:
        self.total_partitions = total_partitions
        self._nodes: List[str] = []  # ordered list of registered node ids
        self._assignment: Dict[str, List[int]] = {}  # node_id → [partition_ids]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def _rebalance(self) -> None:
        """Evenly redistribute all partitions across current nodes."""
        self._assignment.clear()
        if not self._nodes:
            return
        for node_id in self._nodes:
            self._assignment[node_id] = []
        for p in range(self.total_partitions):
            owner = self._nodes[p % len(self._nodes)]
            self._assignment[owner].append(p)

    def register_node(self, node_id: str) -> Dict[str, List[int]]:
        """
        Add a node and rebalance.  Returns the full assignment mapping.
        Idempotent: re-registering an existing node just triggers rebalance.
        """
        if node_id not in self._nodes:
            self._nodes.append(node_id)
            self._nodes.sort()
        self._rebalance()
        return dict(self._assignment)

    def remove_node(self, node_id: str) -> Dict[str, List[int]]:
        """
        Remove a node and rebalance.  Returns the updated assignment.
        """
        if node_id in self._nodes:
            self._nodes.remove(node_id)
        self._rebalance()
        return dict(self._assignment)

    def get_assignment(self) -> Dict[str, List[int]]:
        """Return a copy of the current partition → node mapping."""
        return dict(self._assignment)

    def get_node_for_partition(self, partition_id: int) -> Optional[str]:
        for node_id, partitions in self._assignment.items():
            if partition_id in partitions:
                return node_id
        return None

    def get_node_for_url(self, url: str) -> Optional[str]:
        """Return the node_id that currently owns *url*."""
        partition = url_to_partition(url)
        return self.get_node_for_partition(partition)
