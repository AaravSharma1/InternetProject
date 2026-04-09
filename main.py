"""
CLI entry point for the BFS crawler.

Usage
-----
# Basic BFS crawl
python main.py --seeds https://en.wikipedia.org/wiki/Web_crawler \
               --max-pages 500 --workers 8

# Restrict to a single domain
python main.py --seeds https://news.ycombinator.com \
               --allowed-domains news.ycombinator.com \
               --max-pages 200 --delay 2

# Custom output paths
python main.py --seeds https://example.com \
               --db my_crawl.db --metrics my_run.csv

All options
-----------
  --seeds           One or more seed URLs (space-separated)
  --max-pages       Stop after N successfully fetched pages (default 1000)
  --workers         Max concurrent HTTP requests (default 10)
  --delay           Seconds between requests to the same domain (default 1.0)
  --db              SQLite database path (default crawl.db)
  --metrics         CSV metrics output path (default crawl_metrics.csv)
  --allowed-domains Restrict crawl to these domains (space-separated)
  --bloom-capacity  Expected unique URLs for Bloom sizing (default 1_000_000)
  --log-level       Logging verbosity: DEBUG/INFO/WARNING (default INFO)
"""

import argparse
import asyncio
import logging
import sys

from crawler import BFSFrontier, Crawler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crawler",
        description="Single-node async BFS web crawler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seeds", nargs="+", required=True,
        metavar="URL", help="Seed URL(s) to start the crawl",
    )
    p.add_argument("--max-pages", type=int, default=1_000,
                   help="Maximum pages to fetch")
    p.add_argument("--workers", type=int, default=10,
                   help="Concurrent HTTP workers")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Per-domain request delay in seconds")
    p.add_argument("--db", default="crawl.db",
                   help="SQLite database path")
    p.add_argument("--metrics", default="crawl_metrics.csv",
                   help="CSV metrics output path")
    p.add_argument("--allowed-domains", nargs="*", default=None,
                   metavar="DOMAIN",
                   help="Restrict crawl to these domains")
    p.add_argument("--bloom-capacity", type=int, default=1_000_000,
                   help="Expected unique URLs (tunes Bloom filter)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    crawler = Crawler(
        seed_urls=args.seeds,
        frontier=BFSFrontier(),
        max_pages=args.max_pages,
        max_workers=args.workers,
        rate_limit_delay=args.delay,
        db_path=args.db,
        metrics_path=args.metrics,
        allowed_domains=set(args.allowed_domains) if args.allowed_domains else None,
        bloom_capacity=args.bloom_capacity,
    )

    asyncio.run(crawler.run())
    print(f"\nDone. Crawled {crawler.pages_crawled} pages "
          f"({crawler.pages_relevant} relevant).")
    print(f"  Content store : {args.db}")
    print(f"  Metrics CSV   : {args.metrics}")
    print(f"  Bloom stats   : {crawler.bloom.stats()}")


if __name__ == "__main__":
    main()
