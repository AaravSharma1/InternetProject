#!/usr/bin/env python3
"""
Experiment runner — automates multi-node crawl experiments on localhost.

Spawns the coordinator + N crawler nodes as subprocesses, waits for them
to finish, then collects and merges metrics CSVs.

Experiments:
  scalability   — BFS with 1, 2, 4, 8 nodes (throughput vs. node count)
  compare       — BFS vs link_priority on identical conditions
  comm_overhead — Cross-node bytes exchanged for 2, 4, 8 nodes
"""

import argparse
import csv
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import redis

_HERE = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


def flush_redis(redis_url: str) -> None:
    """Clear all crawl-related keys between experiment runs."""
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    for key in r.scan_iter(match="crawl:*"):
        r.delete(key)
    r.close()
    logger.info("Redis crawl keys flushed")


def start_coordinator(port: int, redis_url: str, log_level: str = "WARNING"):
    cmd = [
        sys.executable, os.path.join(_HERE, "launch.py"),
        "coordinator",
        "--port", str(port),
        "--redis", redis_url,
        "--log-level", log_level,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def start_node(node_id: str, seeds: list, coordinator_url: str,
               redis_url: str, port: int, max_pages: int,
               workers: int, delay: float, mode: str,
               output_dir: str, topic: str = None,
               log_level: str = "INFO"):
    db_path = os.path.join(output_dir, f"{node_id}.db")
    metrics_path = os.path.join(output_dir, f"{node_id}_metrics.csv")
    cmd = [
        sys.executable, os.path.join(_HERE, "launch.py"),
        "node",
        "--node-id", node_id,
        "--seeds", *seeds,
        "--coordinator", coordinator_url,
        "--redis", redis_url,
        "--port", str(port),
        "--max-pages", str(max_pages),
        "--workers", str(workers),
        "--delay", str(delay),
        "--mode", mode,
        "--db", db_path,
        "--metrics", metrics_path,
        "--log-level", log_level,
    ]
    if topic:
        cmd.extend(["--topic", topic])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def merge_metrics(output_dir: str, run_label: str) -> str:
    """Merge per-node CSV files into a single sorted CSV."""
    merged_path = os.path.join(output_dir, f"{run_label}_merged.csv")
    all_rows = []
    header = None
    for csv_file in Path(output_dir).glob("node-*_metrics.csv"):
        node_id = csv_file.stem.replace("_metrics", "")
        with open(csv_file) as f:
            reader = csv.reader(f)
            h = next(reader)
            if header is None:
                header = ["node_id"] + h
            for row in reader:
                all_rows.append([node_id] + row)

    if header is None:
        logger.warning("No node CSV files found in %s — skipping merge", output_dir)
        return merged_path

    all_rows.sort(key=lambda r: float(r[1]))  # sort by timestamp
    with open(merged_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    logger.info("Merged %d rows → %s", len(all_rows), merged_path)
    return merged_path


def run_single_experiment(
    label: str, seeds: list, node_count: int, max_pages: int,
    workers: int, delay: float, mode: str, redis_url: str,
    coord_port: int, base_node_port: int, output_base: str,
    topic: str = None, log_level: str = "INFO",
):
    output_dir = os.path.join(output_base, label)
    os.makedirs(output_dir, exist_ok=True)

    # Flush Redis
    flush_redis(redis_url)

    # Start coordinator
    coordinator_url = f"http://localhost:{coord_port}"
    coord_proc = start_coordinator(coord_port, redis_url, log_level="WARNING")
    time.sleep(1)  # let Flask start

    # Per-node page budget
    pages_per_node = max_pages // node_count

    # Start nodes
    node_procs = []
    for i in range(node_count):
        nid = f"node-{i}"
        port = base_node_port + i
        proc = start_node(
            nid, seeds, coordinator_url, redis_url, port,
            pages_per_node, workers, delay, mode, output_dir,
            topic=topic, log_level=log_level,
        )
        node_procs.append((nid, proc))
        time.sleep(0.5)  # stagger registration

    # Wait for all nodes to finish
    logger.info("Waiting for %d nodes to finish (%s)...", node_count, label)
    for nid, proc in node_procs:
        proc.wait()
        stdout = proc.stdout.read().decode()
        stderr = proc.stderr.read().decode()
        if stdout.strip():
            print(f"[{nid}] {stdout.strip()}")
        if proc.returncode != 0:
            logger.error("[%s] exited with code %d:\n%s", nid,
                         proc.returncode, stderr[-500:])

    # Stop coordinator
    coord_proc.terminate()
    coord_proc.wait(timeout=5)

    # Merge metrics
    merged = merge_metrics(output_dir, label)

    # Compute summary
    total_pages = 0
    total_relevant = 0
    start_time = None
    end_time = None
    try:
        with open(merged) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = float(row["timestamp"])
                if start_time is None or ts < start_time:
                    start_time = ts
                if end_time is None or ts > end_time:
                    end_time = ts
                total_pages += 1
                if row.get("is_relevant", "").lower() == "true":
                    total_relevant += 1
    except Exception:
        pass

    duration = (end_time - start_time) if (start_time and end_time) else 0
    throughput = total_pages / duration if duration > 0 else 0

    summary = {
        "label": label,
        "mode": mode,
        "node_count": node_count,
        "max_pages": max_pages,
        "total_pages": total_pages,
        "total_relevant": total_relevant,
        "duration_sec": round(duration, 2),
        "throughput_pages_per_sec": round(throughput, 2),
        "merged_csv": merged,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment: {label}")
    print(f"  Mode:       {mode}")
    print(f"  Nodes:      {node_count}")
    print(f"  Pages:      {total_pages}")
    print(f"  Duration:   {duration:.1f}s")
    print(f"  Throughput: {throughput:.1f} pages/sec")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")
    return summary


def cmd_scalability(args):
    """Run BFS with varying node counts to measure throughput scaling."""
    results = []
    for n in args.node_counts:
        label = f"scalability_{n}nodes"
        summary = run_single_experiment(
            label=label, seeds=args.seeds, node_count=n,
            max_pages=args.max_pages, workers=args.workers,
            delay=args.delay, mode="bfs", redis_url=args.redis,
            coord_port=args.coord_port, base_node_port=args.base_node_port,
            output_base=args.output,
        )
        results.append(summary)
    # Write combined results
    with open(os.path.join(args.output, "scalability_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Scalability results saved.")


def cmd_compare(args):
    """Run BFS vs link_priority on identical conditions."""
    results = []
    for mode in args.modes:
        label = f"compare_{mode}_{args.nodes}nodes"
        summary = run_single_experiment(
            label=label, seeds=args.seeds, node_count=args.nodes,
            max_pages=args.max_pages, workers=args.workers,
            delay=args.delay, mode=mode, redis_url=args.redis,
            coord_port=args.coord_port, base_node_port=args.base_node_port,
            output_base=args.output, topic=args.topic,
        )
        results.append(summary)
    with open(os.path.join(args.output, "compare_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Comparison results saved.")


def cmd_comm_overhead(args):
    """Measure cross-node communication for varying node counts."""
    results = []
    for n in args.node_counts:
        label = f"comm_{n}nodes"
        summary = run_single_experiment(
            label=label, seeds=args.seeds, node_count=n,
            max_pages=args.max_pages, workers=args.workers,
            delay=args.delay, mode="bfs", redis_url=args.redis,
            coord_port=args.coord_port, base_node_port=args.base_node_port,
            output_base=args.output,
        )
        # Also pull comm stats from Redis before flushing
        r = redis.Redis.from_url(args.redis, decode_responses=True)
        comm_data = {}
        for key in r.scan_iter(match="crawl:comm:*"):
            val = r.get(key)
            comm_data[key if isinstance(key, str) else key.decode()] = int(val or 0)
        r.close()
        summary["comm_stats"] = comm_data
        results.append(summary)
    with open(os.path.join(args.output, "comm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Communication overhead results saved.")


def main():
    parser = argparse.ArgumentParser(description="Distributed crawler experiments")
    parser.add_argument("--redis", default="redis://localhost:6379")
    parser.add_argument("--coord-port", type=int, default=5000)
    parser.add_argument("--base-node-port", type=int, default=6001)
    parser.add_argument("--output", default="experiments")
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="experiment", required=True)

    # scalability
    s = sub.add_parser("scalability")
    s.add_argument("--seeds", nargs="+", required=True)
    s.add_argument("--max-pages", type=int, default=5000)
    s.add_argument("--workers", type=int, default=10)
    s.add_argument("--delay", type=float, default=0.5)
    s.add_argument("--node-counts", nargs="+", type=int, default=[1, 2, 4, 8])

    # compare
    c = sub.add_parser("compare")
    c.add_argument("--seeds", nargs="+", required=True)
    c.add_argument("--max-pages", type=int, default=5000)
    c.add_argument("--workers", type=int, default=10)
    c.add_argument("--delay", type=float, default=0.5)
    c.add_argument("--nodes", type=int, default=4)
    c.add_argument("--modes", nargs="+", default=["bfs", "link_priority"])
    c.add_argument("--topic", default=None)

    # comm_overhead
    co = sub.add_parser("comm_overhead")
    co.add_argument("--seeds", nargs="+", required=True)
    co.add_argument("--max-pages", type=int, default=5000)
    co.add_argument("--workers", type=int, default=10)
    co.add_argument("--delay", type=float, default=0.5)
    co.add_argument("--node-counts", nargs="+", type=int, default=[2, 4, 8])

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if args.experiment == "scalability":
        cmd_scalability(args)
    elif args.experiment == "compare":
        cmd_compare(args)
    elif args.experiment == "comm_overhead":
        cmd_comm_overhead(args)


if __name__ == "__main__":
    main()
