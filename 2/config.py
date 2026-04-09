"""
Shared constants for the distributed crawler (Part 2).

All Redis key prefixes, default ports, heartbeat parameters, and Bloom
filter parameters are centralised here so every module agrees on naming.
"""

import math

# ── Network defaults ────────────────────────────────────────────────
COORDINATOR_DEFAULT_PORT = 5000
NODE_DEFAULT_PORT_BASE = 6000
REDIS_URL = "redis://localhost:6379"

# ── Heartbeat ───────────────────────────────────────────────────────
HEARTBEAT_INTERVAL = 10          # seconds between heartbeats
HEARTBEAT_MISS_THRESHOLD = 3     # missed beats before node is declared dead

# ── Hash partitioning ──────────────────────────────────────────────
TOTAL_PARTITIONS = 256           # virtual partitions for URL space

# ── Bloom filter (must match Part 1 defaults) ──────────────────────
BLOOM_CAPACITY = 1_000_000
BLOOM_ERROR_RATE = 0.01
BLOOM_BIT_SIZE = max(1, int(-BLOOM_CAPACITY * math.log(BLOOM_ERROR_RATE)
                             / (math.log(2) ** 2)))
BLOOM_NUM_HASHES = max(1, int((BLOOM_BIT_SIZE / BLOOM_CAPACITY)
                               * math.log(2)))

# ── Redis key schema ───────────────────────────────────────────────
BLOOM_REDIS_KEY = "crawl:bloom"
NODE_CHANNEL_PREFIX = "crawl:node:"          # pub/sub: crawl:node:<id>
ASSIGNMENT_KEY = "crawl:assignment"          # JSON of current mapping
ASSIGNMENT_CHANNEL = "crawl:assignment_updates"  # pub/sub for rebalance
NODE_REGISTRY_KEY = "crawl:nodes"           # Redis hash {node_id: json}
COMM_STATS_PREFIX = "crawl:comm:"           # crawl:comm:<id>:<min_ts>:sent/recv
