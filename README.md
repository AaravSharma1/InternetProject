# AI-Based Distributed Web Crawler with Semantic Ranking

Three-tier distributed web crawler that compares BFS, link-priority, and embedding-based semantic crawl strategies. Runs 2–8 nodes in parallel, coordinated by a Flask service over Redis.

## Architecture

```
Tier 1 — Coordinator (Flask)
  • Node registration + consistent hash-ring partitioning (256 virtual partitions)
  • Heartbeat monitoring; auto-rebalances partitions when a node crashes
  • Stores current assignment in Redis so nodes can resync after restarts

Tier 2 — Crawler Nodes (2–8 nodes, each running independently)
  Frontier → Fetcher → Parser → Bloom Filter → Router
  • Frontier: pluggable priority queue (BFS FIFO, link-count heap, or semantic score heap)
  • Fetcher: async aiohttp with robots.txt compliance, per-domain rate limiting, exponential backoff
  • Parser: BeautifulSoup/lxml — extracts anchor text + ±200-char context window per link
  • Bloom filter: local in-memory + shared Redis bitmap (atomic Lua SETBIT/GETBIT)
  • Router: hashes each discovered URL, forwards off-partition URLs to the owning node via Redis pub/sub

Tier 3 — Redis
  • Distributed Bloom filter (shared across all nodes)
  • Pub/sub channels for lateral URL forwarding between nodes
  • Live stats keys read by the dashboard

Semantic Module (optional, plugs into any node)
  • all-MiniLM-L6-v2 via sentence-transformers (~80 MB, <10 ms/URL on CPU)
  • Scores each link by cosine similarity to a topic centroid before it enters the frontier
  • Centroid drifts as relevant pages are found: centroid = 0.95·old + 0.05·new_embedding
  • Three modes: bfs | link_priority | semantic
```

## Repository layout

```
1/          Single-node crawler engine (base classes used by everything)
2/          Distributed layer: coordinator, nodes, run_experiment.py
3/          Semantic prioritizer + relevance classifier
demos_and_evals/   Eval pipeline, live dashboard, figures
```

## Quick start

### 1. Install dependencies

```bash
pip install -r 1/requirements.txt
pip install -r 2/requirements.txt
pip install -r 3/requirements.txt
pip install pandas flask-cors lxml
```

### 2. Start Redis

```bash
brew install redis   # macOS (one-time)
brew services start redis
```

### 3. Run all 3 crawl modes

`run_experiment.py` starts the coordinator and nodes automatically, flushes Redis between modes, and merges per-node CSVs.

> **Note:** macOS uses port 5000 for AirPlay. Always pass `--coord-port 5001`.

```bash
python 2/run_experiment.py --output my_experiments/ --coord-port 5001 compare \
  --seeds https://arxiv.org/list/cs.LG/recent \
  --modes bfs link_priority semantic \
  --topic "machine learning research papers" \
  --nodes 2 \
  --max-pages 150 \
  --workers 6 \
  --delay 1.0
```

This runs each mode sequentially (~5–10 min total) and produces:

```
my_experiments/
  compare_bfs_2nodes/           compare_bfs_2nodes_merged.csv
  compare_link_priority_2nodes/ compare_link_priority_2nodes_merged.csv
  compare_semantic_2nodes/      compare_semantic_2nodes_merged.csv
```

### 4. Run the eval pipeline

```bash
python demos_and_evals/eval_pipeline.py \
  --bfs  my_experiments/compare_bfs_2nodes/compare_bfs_2nodes_merged.csv \
  --link my_experiments/compare_link_priority_2nodes/compare_link_priority_2nodes_merged.csv \
  --sem  my_experiments/compare_semantic_2nodes/compare_semantic_2nodes_merged.csv \
  --output my_results/
```

Outputs: `summary_table.csv`, `relevance_at_depth.png`, `bandwidth_efficiency.png`, `throughput_bar.png`, `redundancy_bar.png`.

### 5. Live dashboard (run alongside step 3)

```bash
python demos_and_evals/dashboard_backend.py
# Open http://localhost:5050
```

Reads live stats from Redis — shows pages/sec per node, harvest rate over time, and relevance score histogram. Falls back to simulated data if Redis is unreachable.

## Single-node usage (no Redis required)

```bash
# Plain BFS
python 1/main.py --seeds https://arxiv.org/list/cs.LG/recent --max-pages 500

# Topic-focused crawl (semantic scoring + priority frontier)
python 1/main.py --seeds https://arxiv.org/list/cs.LG/recent \
  --topic "deep learning research papers" \
  --topic-threshold 0.3 \
  --max-pages 500
```

## run_experiment.py flags

Global flags come **before** the subcommand:

```
--output DIR          Output directory (default: experiments/)
--coord-port PORT     Coordinator port (default: 5000; use 5001 on macOS)
--redis URL           Redis URL (default: redis://localhost:6379)
--log-level LEVEL     DEBUG / INFO / WARNING (default: INFO)
```

Subcommand `compare` flags:

```
--seeds URL [URL ...]  Seed URLs
--modes MODE [...]     bfs, link_priority, semantic (can combine all three)
--topic TEXT           Topic phrase for semantic mode
--nodes N              Number of crawler nodes per mode (default: 4)
--max-pages N          Total pages per mode across all nodes (default: 5000)
--workers N            Concurrent HTTP workers per node (default: 10)
--delay SEC            Per-domain rate limit delay (default: 0.5)
```

## Outputs

**Per-node `*_metrics.csv`** — one row per fetched URL:

| Column | Description |
|---|---|
| `timestamp` | Unix epoch |
| `url` | Fetched URL |
| `bytes_downloaded` | Raw bytes |
| `fetch_latency_ms` | Wall-clock fetch time |
| `status_code` | HTTP status (0 on failure) |
| `is_relevant` | Scored by relevance function |
| `cumulative_pages` | Running total for this node |
| `cumulative_relevant` | Running relevant total |

**Merged CSV** — same schema with `node_id` prepended, sorted by timestamp.

**`*.db`** — SQLite with `pages` and `links` tables per node.

## Components

### Coordinator (`2/coordinator/`)
Flask service. Registers nodes, assigns hash-ring partitions, monitors heartbeats every 10 s, and auto-rebalances when a node is declared dead (3 missed heartbeats).

### DistributedCrawler (`2/node/distributed_crawler.py`)
Subclasses the single-node `Crawler`. Overrides `_process()` to route discovered links by URL hash: local links go to the local frontier, off-partition links are forwarded via Redis pub/sub. Overrides `_drain()` with a 5-second grace period to handle URLs in transit.

### SemanticPrioritizer (`3/semantic_prioritizer.py`)
Loads `all-MiniLM-L6-v2`. Scores each `ExtractedLink` (anchor text + ±200-char context + URL path tokens) by cosine similarity to a topic centroid. Updates the centroid via EMA after each relevant page.

### Eval pipeline (`demos_and_evals/eval_pipeline.py`)
Reads merged CSVs, computes harvest rate, relevance-at-depth, bandwidth efficiency, throughput, MinHash redundancy, and cross-node overhead. Pass `--demo` for synthetic data without running a crawl.
