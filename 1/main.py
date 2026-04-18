"""
CLI entry point for the BFS crawler.

Usage
-----
# Basic BFS crawl (no topic — accepts all pages)
python main.py --seeds https://en.wikipedia.org/wiki/Web_crawler \
               --max-pages 500 --workers 8

# Topic-focused crawl (semantic prioritization + relevance filtering)
python main.py --seeds https://arxiv.org \
               --topic "deep learning research papers" "neural networks survey" \
               --topic-threshold 0.3 \
               --max-pages 500

# Restrict to a single domain
python main.py --seeds https://news.ycombinator.com \
               --allowed-domains news.ycombinator.com \
               --max-pages 200 --delay 2

All options
-----------
  --seeds           One or more seed URLs (space-separated)
  --topic           One or more topic description phrases for focused crawling.
                    When provided, uses SemanticPrioritizer to score pages and
                    rank discovered links.  Omit for plain BFS.
  --topic-threshold Cosine similarity cutoff for marking a page relevant (default 0.3)
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
from pathlib import Path

from crawler import BFSFrontier, Crawler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crawler",
        description="Single-node async web crawler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seeds", nargs="+", required=True,
        metavar="URL", help="Seed URL(s) to start the crawl",
    )
    p.add_argument(
        "--topic", nargs="+", default=None,
        metavar="PHRASE",
        help="Topic description phrases for semantic focused crawling",
    )
    p.add_argument(
        "--topic-threshold", type=float, default=0.3,
        help="Cosine similarity cutoff to mark a page as relevant (0–1)",
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


def _build_topic_hooks(topic_phrases, threshold):
    """Return (frontier, relevance_fn, link_priority_fn) for semantic crawling."""
    part3_dir = str(Path(__file__).resolve().parent.parent / "3")
    if part3_dir not in sys.path:
        sys.path.insert(0, part3_dir)

    from semantic_prioritizer import SemanticPrioritizer
    from crawler import PriorityFrontier

    logging.getLogger(__name__).info(
        "Loading SemanticPrioritizer (all-MiniLM-L6-v2)..."
    )
    prioritizer = SemanticPrioritizer(relevance_threshold=threshold)
    centroid = prioritizer.init_centroid(topic_phrases)

    logging.getLogger(__name__).info(
        "Topic centroid initialized from %d phrase(s): %s",
        len(topic_phrases), topic_phrases,
    )

    def relevance_fn(page):
        score = prioritizer.score(page.text[:1000], centroid)
        return score >= threshold

    def link_priority_fn(link):
        # PriorityFrontier is a min-heap: negate so higher scores surface first
        ctx = prioritizer.build_url_context(link.anchor_text, link.context, link.url)
        return -prioritizer.score(ctx, centroid)

    return PriorityFrontier(), relevance_fn, link_priority_fn


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    frontier = BFSFrontier()
    relevance_fn = None
    link_priority_fn = None

    if args.topic:
        frontier, relevance_fn, link_priority_fn = _build_topic_hooks(
            args.topic, args.topic_threshold
        )

    crawler = Crawler(
        seed_urls=args.seeds,
        frontier=frontier,
        max_pages=args.max_pages,
        max_workers=args.workers,
        rate_limit_delay=args.delay,
        db_path=args.db,
        metrics_path=args.metrics,
        allowed_domains=set(args.allowed_domains) if args.allowed_domains else None,
        bloom_capacity=args.bloom_capacity,
        relevance_fn=relevance_fn,
        link_priority_fn=link_priority_fn,
    )

    asyncio.run(crawler.run())
    print(f"\nDone. Crawled {crawler.pages_crawled} pages "
          f"({crawler.pages_relevant} relevant).")
    print(f"  Content store : {args.db}")
    print(f"  Metrics CSV   : {args.metrics}")
    print(f"  Bloom stats   : {crawler.bloom.stats()}")


if __name__ == "__main__":
    main()
