# InternetProject — Web Crawler Engine

Single-node async BFS web crawler built with Python `asyncio` + `aiohttp`. This is the foundational engine that all experiments plug into.

## Structure

```
1/
├── crawler/
│   ├── __init__.py      # Public API surface
│   ├── bloom.py         # Bloom filter for URL deduplication
│   ├── crawler.py       # Main orchestrator (BFS crawl loop)
│   ├── fetcher.py       # Async HTTP fetcher (robots.txt, rate limiting)
│   ├── frontier.py      # Pluggable URL frontier (BFS + priority queue)
│   ├── metrics.py       # CSV metrics logger
│   ├── parser.py        # HTML parser + link extractor
│   └── store.py         # SQLite content store
├── main.py              # CLI entry point
└── requirements.txt
```

## Setup

```bash
pip install -r 1/requirements.txt
```

## Usage

```bash
# Basic BFS crawl
python 1/main.py --seeds https://en.wikipedia.org/wiki/Web_crawler --max-pages 500

# Restrict to one domain, 8 workers, 2s delay
python 1/main.py --seeds https://news.ycombinator.com \
                 --allowed-domains news.ycombinator.com \
                 --max-pages 200 --workers 8 --delay 2

# Custom output paths
python 1/main.py --seeds https://example.com --db run1.db --metrics run1.csv
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--seeds` | *(required)* | One or more seed URLs |
| `--max-pages` | 1000 | Stop after N fetched pages |
| `--workers` | 10 | Max concurrent HTTP requests |
| `--delay` | 1.0 | Per-domain request delay (seconds) |
| `--db` | `crawl.db` | SQLite output path |
| `--metrics` | `crawl_metrics.csv` | CSV metrics output path |
| `--allowed-domains` | *(none)* | Restrict crawl to these domains |
| `--bloom-capacity` | 1,000,000 | Expected unique URLs (tunes Bloom filter) |
| `--log-level` | INFO | DEBUG / INFO / WARNING / ERROR |

## Outputs

**`crawl.db`** — SQLite database with two tables:

- `pages(url, fetch_time, status_code, byte_size, title, html, text, error)`
- `links(src_url, dst_url, anchor_text, context)` — each link includes a ~200-char text window around the `<a>` tag

**`crawl_metrics.csv`** — one row per fetched URL:

| Column | Description |
|---|---|
| `timestamp` | Unix epoch at fetch start |
| `url` | Fetched URL |
| `bytes_downloaded` | Raw bytes received |
| `fetch_latency_ms` | Wall-clock fetch time |
| `status_code` | HTTP status (0 on connection failure) |
| `is_relevant` | Set by `relevance_fn` (default: always `True`) |
| `cumulative_pages` | Running total of fetched pages |
| `cumulative_relevant` | Running total of relevant pages |
| `error` | Error message if fetch failed |

## Plugging in experiment code

The frontier and relevance function are both injectable — no changes to engine code required:

```python
from crawler import Crawler, PriorityFrontier

# Swap in a priority frontier (e.g. for link-count or semantic scoring)
frontier = PriorityFrontier()

# Provide a relevance function (receives a ParsedPage, returns bool)
def is_relevant(page):
    return "machine learning" in page.text.lower()

crawler = Crawler(
    seed_urls=["https://example.com"],
    frontier=frontier,
    relevance_fn=is_relevant,
    max_pages=1000,
)

import asyncio
asyncio.run(crawler.run())
```

`PriorityFrontier.push(url, priority=score)` accepts any float — lower value is dequeued first.

## Components

### Bloom Filter (`bloom.py`)
Pure-Python implementation using `k` MD5-based hash functions. Optimal bit array size and hash count are derived from `capacity` and `error_rate`. Reports `estimated_fpr` and fill ratio at any point via `.stats()`.

### URL Frontier (`frontier.py`)
- **`BFSFrontier`** — FIFO via `asyncio.Queue`. Implements BFS and is the default.
- **`PriorityFrontier`** — min-heap; lower priority value dequeued first. Thread-safe via `asyncio.Lock`.

Both expose `push(url, priority, metadata)` / `pop()` / `empty()`.

### Fetcher (`fetcher.py`)
- Checks `robots.txt` per origin (cached, fetched async).
- Enforces a configurable per-domain delay via `asyncio.Lock`.
- Streams response body with a 5 MB cap; skips non-HTML content types.
- Retries transient errors with exponential back-off.

### Parser (`parser.py`)
Uses `lxml` via BeautifulSoup. For each `<a>` tag, walks up the DOM to the nearest block-level ancestor and slices a ±200-char window around the anchor text for context.

### Content Store (`store.py`)
SQLite with WAL mode. `save_page()` upserts; `save_links()` ignores duplicates. Both `pages` and `links` are queryable for downstream analysis.

### Metrics Logger (`metrics.py`)
Append-only CSV flushed after every row — a partial log is always valid even if the crawl crashes.
