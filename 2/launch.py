#!/usr/bin/env python3
"""
CLI entry point for the distributed crawler.

Usage:
    # Start the coordinator
    python launch.py coordinator [--port 5000] [--redis redis://localhost:6379]

    # Start a crawler node
    python launch.py node --node-id node-0 --seeds https://example.com \
        [--coordinator http://localhost:5000] [--redis redis://localhost:6379] \
        [--port 6001] [--max-pages 1000] [--workers 10] [--delay 1.0] \
        [--mode bfs|link_priority|semantic] [--topic "machine learning"]
"""

import argparse
import asyncio
import logging
import os
import sys

# ── Path setup ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "1"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "3"))
sys.path.insert(0, _HERE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distributed web crawler — coordinator or node")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── coordinator ─────────────────────────────────────────────────
    coord = sub.add_parser("coordinator", help="Start the coordinator service")
    coord.add_argument("--port", type=int, default=5000)
    coord.add_argument("--redis", default="redis://localhost:6379")
    coord.add_argument("--log-level", default="INFO")

    # ── node ────────────────────────────────────────────────────────
    node = sub.add_parser("node", help="Start a crawler node")
    node.add_argument("--node-id", required=True)
    node.add_argument("--seeds", nargs="+", required=True)
    node.add_argument("--coordinator", default="http://localhost:5000")
    node.add_argument("--redis", default="redis://localhost:6379")
    node.add_argument("--port", type=int, default=6001,
                      help="Node's own port (for coordinator registration)")
    node.add_argument("--max-pages", type=int, default=1000)
    node.add_argument("--workers", type=int, default=10)
    node.add_argument("--delay", type=float, default=1.0,
                      help="Rate limit delay per domain (seconds)")
    node.add_argument("--mode", choices=["bfs", "link_priority", "semantic"],
                      default="bfs")
    node.add_argument("--topic", default=None,
                      help="Topic description for semantic mode")
    node.add_argument("--db", default=None,
                      help="SQLite DB path (default: <node-id>.db)")
    node.add_argument("--metrics", default=None,
                      help="Metrics CSV path (default: <node-id>_metrics.csv)")
    node.add_argument("--bloom-capacity", type=int, default=1_000_000)
    node.add_argument("--allowed-domains", nargs="*", default=None)
    node.add_argument("--log-level", default="INFO")

    return parser


def run_coordinator(args) -> None:
    from coordinator.app import create_app
    app = create_app(redis_url=args.redis)
    app.run(host="0.0.0.0", port=args.port)


def run_node(args) -> None:
    from node.distributed_crawler import DistributedCrawler

    allowed = set(args.allowed_domains) if args.allowed_domains else None

    crawler = DistributedCrawler(
        seed_urls=args.seeds,
        node_id=args.node_id,
        coordinator_url=args.coordinator,
        redis_url=args.redis,
        mode=args.mode,
        topic=args.topic,
        max_pages=args.max_pages,
        max_workers=args.workers,
        rate_limit_delay=args.delay,
        db_path=args.db,
        metrics_path=args.metrics,
        bloom_capacity=args.bloom_capacity,
        allowed_domains=allowed,
        node_port=args.port,
    )
    asyncio.run(crawler.run())

    # Print summary
    print(f"\n{'='*60}")
    print(f"Node {args.node_id} finished")
    print(f"  Mode:            {args.mode}")
    print(f"  Pages crawled:   {crawler.pages_crawled}")
    print(f"  Pages relevant:  {crawler.pages_relevant}")
    print(f"  URLs received:   {crawler.receiver.urls_received if crawler.receiver else 0}")
    print(f"  Bytes sent:      {crawler.comm_tracker.total_sent:,}")
    print(f"  Bytes received:  {crawler.comm_tracker.total_received:,}")
    print(f"  Content store:   {crawler.db_path}")
    print(f"  Metrics CSV:     {crawler.metrics_path}")
    print(f"{'='*60}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    )

    if args.command == "coordinator":
        run_coordinator(args)
    elif args.command == "node":
        run_node(args)


if __name__ == "__main__":
    main()
