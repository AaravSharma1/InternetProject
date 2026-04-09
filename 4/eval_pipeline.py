"""
eval_pipeline.py — Evaluation Pipeline for Semantic Web Crawler
Person 4: Evaluation, Visualization & Demo

Usage:
    python eval_pipeline.py --log crawl_log.csv --mode all --output results/
    python eval_pipeline.py --log crawl_log.csv --mode semantic --redundancy
    python eval_pipeline.py --demo   # runs on synthetic data for testing

Reads the shared crawl log CSV (schema: timestamp, node_id, url, bytes,
fetch_ms, relevance_score, is_relevant, mode) and produces:
  - Summary table (all metrics across BFS / Link-Priority / Semantic)
  - Relevance-at-depth line chart
  - Bandwidth efficiency curve
  - Throughput scaling bar chart
  - Redundancy rate per mode
  - Per-run mean ± std tables

Dependencies:
    pip install pandas matplotlib datasketch numpy scikit-learn
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datasketch import MinHash, MinHashLSH

# ─────────────────────────────────────────────
# Schema Contract (matches Person 1's CSV spec)
# ─────────────────────────────────────────────
REQUIRED_COLS = {
    "timestamp", "node_id", "url", "bytes",
    "fetch_ms", "relevance_score", "is_relevant", "mode"
}
# Optional column used when content text is present for MinHash
OPTIONAL_TEXT_COL = "page_text"

MODES = ["bfs", "link_priority", "semantic"]
RELEVANCE_THRESHOLD = 0.5   # is_relevant flag must agree; used as fallback
JACCARD_THRESHOLD = 0.8     # near-duplicate detection cutoff
MINHASH_NUM_PERM = 128
SHINGLE_SIZE = 5            # 5-word shingles


# ─────────────────────────────────────────────
# 1. Load & Validate
# ─────────────────────────────────────────────

def load_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Log is missing columns: {missing}")
    df["is_relevant"] = df["is_relevant"].astype(bool)
    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce").fillna(0)
    df["mode"] = df["mode"].str.lower().str.strip()
    df = df.sort_values(["mode", "timestamp"]).reset_index(drop=True)
    return df


def make_synthetic_log(n_pages: int = 5000, runs: int = 3) -> pd.DataFrame:
    """Generate synthetic data so the pipeline can be tested standalone."""
    rng = np.random.default_rng(42)
    rows = []
    t0 = pd.Timestamp("2025-01-01")
    for run in range(1, runs + 1):
        for mode, rel_rate in [("bfs", 0.18), ("link_priority", 0.32), ("semantic", 0.61)]:
            # Semantic mode front-loads relevant pages
            if mode == "semantic":
                probs = np.linspace(0.9, 0.3, n_pages)
                probs /= probs.sum()
            else:
                probs = np.ones(n_pages) / n_pages
            for i in range(n_pages):
                is_rel = rng.random() < (rel_rate * (2 - probs[i] * n_pages / rel_rate if mode == "semantic" else 1))
                is_rel = bool(is_rel)
                rows.append({
                    "timestamp": t0 + pd.Timedelta(seconds=i * 0.3),
                    "node_id": f"node_{rng.integers(1, 5)}",
                    "url": f"https://example.com/page/{mode}/{i}_{run}",
                    "bytes": int(rng.integers(5_000, 200_000)),
                    "fetch_ms": int(rng.integers(50, 800)),
                    "relevance_score": float(rng.uniform(0.6, 0.99) if is_rel else rng.uniform(0.01, 0.45)),
                    "is_relevant": is_rel,
                    "mode": mode,
                    "run": run,
                    "page_text": f"sample text for page {i} in mode {mode} run {run} " * 20,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 2. MinHash Redundancy Detection
# ─────────────────────────────────────────────

def shinglize(text: str, k: int = SHINGLE_SIZE) -> set:
    words = text.lower().split()
    return {" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def compute_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    for shingle in shinglize(text):
        m.update(shingle.encode("utf8"))
    return m


def compute_redundancy_rate(df: pd.DataFrame) -> dict[str, float]:
    """
    For each mode, compute near-duplicate rate using MinHash LSH.
    Returns {mode: redundancy_rate_0_to_1}.
    Requires 'page_text' column; if absent, estimates from URL similarity.
    """
    results = {}
    has_text = OPTIONAL_TEXT_COL in df.columns

    for mode, group in df.groupby("mode"):
        lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=MINHASH_NUM_PERM)
        duplicates = 0
        total = 0
        for idx, row in group.iterrows():
            text = row[OPTIONAL_TEXT_COL] if has_text else row["url"]
            mh = compute_minhash(str(text))
            key = str(idx)
            try:
                neighbors = lsh.query(mh)
                if neighbors:
                    duplicates += 1
                else:
                    lsh.insert(key, mh)
            except Exception:
                lsh.insert(key, mh)
            total += 1
        results[mode] = duplicates / total if total > 0 else 0.0
        print(f"  [{mode}] redundancy={results[mode]:.2%}  ({duplicates}/{total} near-dupes)")
    return results


# ─────────────────────────────────────────────
# 3. Core Metrics
# ─────────────────────────────────────────────

def harvest_rate(df: pd.DataFrame) -> dict:
    return df.groupby("mode")["is_relevant"].mean().to_dict()


def relevance_at_depth(df: pd.DataFrame, sample_points: int = 100) -> dict:
    """
    Cumulative relevance rate as pages are crawled (in discovery order).
    Returns {mode: (x_array, y_array)} where x is page number, y is cum. rate.
    """
    result = {}
    for mode, group in df.groupby("mode"):
        group = group.reset_index(drop=True)
        cum_rel = group["is_relevant"].cumsum()
        x = np.linspace(0, len(group) - 1, sample_points, dtype=int)
        y = (cum_rel.iloc[x] / (x + 1)).values
        result[mode] = (x, y)
    return result


def bandwidth_efficiency(df: pd.DataFrame) -> dict:
    """Cumulative MB downloaded per relevant page discovered."""
    result = {}
    for mode, group in df.groupby("mode"):
        group = group.reset_index(drop=True)
        cum_bytes = group["bytes"].cumsum() / 1e6  # → MB
        cum_relevant = group["is_relevant"].cumsum().replace(0, np.nan)
        ratio = cum_bytes / cum_relevant
        result[mode] = ratio.values
    return result


def throughput_per_node(df: pd.DataFrame) -> dict:
    """Pages/sec per node. Approximated from fetch_ms."""
    result = {}
    for mode, group in df.groupby("mode"):
        nodes = group["node_id"].nunique()
        total_time_sec = (group["timestamp"].max() - group["timestamp"].min()).total_seconds()
        if total_time_sec <= 0:
            total_time_sec = len(group) * 0.3
        pps = len(group) / max(total_time_sec, 1) / max(nodes, 1)
        result[mode] = round(pps, 2)
    return result


def cross_node_bytes(df: pd.DataFrame) -> dict:
    """
    Estimates cross-node communication as forwarded URLs * avg URL size (100 bytes).
    Exact measurement requires Per 2's Redis logs; this is the fallback estimator.
    """
    result = {}
    URL_FORWARD_BYTES = 100
    for mode, group in df.groupby("mode"):
        n_nodes = group["node_id"].nunique()
        # URLs that would be forwarded ≈ URLs not "owned" by the fetching node
        # Proxy: pages where node_id hash % n_nodes != expected_shard
        forwarded = len(group) * (1 - 1 / max(n_nodes, 1)) * 0.5
        result[mode] = int(forwarded * URL_FORWARD_BYTES)
    return result


def mean_std_across_runs(df: pd.DataFrame, metric_fn) -> pd.DataFrame:
    """Compute mean ± std of a scalar metric across runs (requires 'run' column)."""
    if "run" not in df.columns:
        return pd.DataFrame()
    records = []
    for run in df["run"].unique():
        sub = df[df["run"] == run]
        vals = metric_fn(sub)
        records.append(vals)
    keys = records[0].keys()
    rows = {}
    for k in keys:
        vals = [r[k] for r in records if k in r]
        rows[k] = {"mean": np.mean(vals), "std": np.std(vals)}
    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────
# 4. Plotting
# ─────────────────────────────────────────────

COLORS = {"bfs": "#888780", "link_priority": "#378ADD", "semantic": "#1D9E75"}
DASHES = {"bfs": (6, 2), "link_priority": (3, 1), "semantic": None}
LABELS = {"bfs": "BFS baseline", "link_priority": "Link-Priority", "semantic": "Semantic (ours)"}

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D3D1C7")
    ax.spines["bottom"].set_color("#D3D1C7")
    ax.tick_params(colors="#5F5E5A", labelsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#2C2C2A", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#5F5E5A")
    ax.set_ylabel(ylabel, fontsize=9, color="#5F5E5A")
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.5, linestyle="--")


def plot_relevance_at_depth(rad: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for mode, (x, y) in rad.items():
        ls = "--" if DASHES[mode] else "-"
        ax.plot(x, y * 100, color=COLORS[mode], linestyle=ls, linewidth=1.8,
                label=LABELS[mode], marker="o" if mode == "semantic" else None,
                markevery=10, markersize=3)
    _style_ax(ax, "Relevance at depth", "Pages crawled", "Cumulative harvest rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_bandwidth_efficiency(bwe: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for mode, vals in bwe.items():
        ls = "--" if DASHES[mode] else "-"
        ax.plot(np.linspace(0, len(vals), len(vals)), vals,
                color=COLORS[mode], linestyle=ls, linewidth=1.8, label=LABELS[mode])
    _style_ax(ax, "Bandwidth efficiency", "Pages crawled", "MB per relevant page")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_throughput_bar(tpn: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    modes = list(tpn.keys())
    vals = [tpn[m] for m in modes]
    bars = ax.bar([LABELS[m] for m in modes], vals,
                  color=[COLORS[m] for m in modes], width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#2C2C2A")
    _style_ax(ax, "Throughput per node", "", "Pages/sec/node")
    ax.set_xticklabels([LABELS[m] for m in modes], fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_redundancy_bar(redundancy: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    modes = list(redundancy.keys())
    vals = [redundancy[m] * 100 for m in modes]
    bars = ax.bar([LABELS[m] for m in modes], vals,
                  color=[COLORS[m] for m in modes], width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color="#2C2C2A")
    _style_ax(ax, "Near-duplicate rate by mode", "", "Near-duplicate pages (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xticklabels([LABELS[m] for m in modes], fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────
# 5. Summary Table
# ─────────────────────────────────────────────

def build_summary_table(df: pd.DataFrame, redundancy: dict) -> pd.DataFrame:
    hr = harvest_rate(df)
    tpn = throughput_per_node(df)
    cnb = cross_node_bytes(df)
    rows = []
    for mode in MODES:
        if mode not in df["mode"].values:
            continue
        sub = df[df["mode"] == mode]
        avg_bytes_per_relevant = (
            sub["bytes"].sum() / sub["is_relevant"].sum()
            if sub["is_relevant"].sum() > 0 else float("inf")
        )
        rows.append({
            "Mode": LABELS.get(mode, mode),
            "Harvest Rate (%)": f"{hr.get(mode, 0) * 100:.1f}",
            "Avg MB/Relevant": f"{avg_bytes_per_relevant / 1e6:.2f}",
            "Redundancy (%)": f"{redundancy.get(mode, 0) * 100:.1f}",
            "Throughput (pg/s/node)": f"{tpn.get(mode, 0):.2f}",
            "Cross-Node (KB)": f"{cnb.get(mode, 0) / 1e3:.1f}",
            "Pages Crawled": len(sub),
        })
    return pd.DataFrame(rows).set_index("Mode")


# ─────────────────────────────────────────────
# 6. Main Entry Point
# ─────────────────────────────────────────────

def run_pipeline(log_path: str | None, output_dir: str, demo: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Semantic Crawler Evaluation Pipeline ===\n")

    if demo or log_path is None:
        print("Running on synthetic demo data...")
        df = make_synthetic_log(n_pages=3000, runs=3)
    else:
        print(f"Loading log: {log_path}")
        df = load_log(log_path)

    print(f"Loaded {len(df):,} records | modes: {df['mode'].unique().tolist()}\n")

    # --- Redundancy (most expensive step) ---
    print("[1/5] Computing MinHash redundancy rates...")
    redundancy = compute_redundancy_rate(df)

    # --- Core metrics ---
    print("\n[2/5] Computing harvest rates, throughput, bandwidth...")
    rad = relevance_at_depth(df)
    bwe = bandwidth_efficiency(df)
    tpn = throughput_per_node(df)

    # --- Summary table ---
    print("\n[3/5] Building summary table...")
    summary = build_summary_table(df, redundancy)
    table_path = os.path.join(output_dir, "summary_table.csv")
    summary.to_csv(table_path)
    print(f"\n{'='*58}")
    print(summary.to_string())
    print(f"{'='*58}\n  Saved: {table_path}\n")

    # --- Plots ---
    print("[4/5] Generating figures...")
    plot_relevance_at_depth(rad, os.path.join(output_dir, "relevance_at_depth.png"))
    plot_bandwidth_efficiency(bwe, os.path.join(output_dir, "bandwidth_efficiency.png"))
    plot_throughput_bar(tpn, os.path.join(output_dir, "throughput_bar.png"))
    plot_redundancy_bar(redundancy, os.path.join(output_dir, "redundancy_bar.png"))

    # --- Mean ± std across runs ---
    if "run" in df.columns:
        print("\n[5/5] Computing mean ± std across runs...")
        stats = mean_std_across_runs(df, harvest_rate)
        stats_path = os.path.join(output_dir, "harvest_rate_stats.csv")
        stats.to_csv(stats_path)
        print(f"  Harvest rate mean±std:\n{stats.to_string()}")
        print(f"  Saved: {stats_path}")
    else:
        print("[5/5] Skipped run stats (no 'run' column in log).")

    print("\n=== Pipeline complete ===")
    print(f"All outputs written to: {output_dir}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Crawler Evaluation Pipeline")
    parser.add_argument("--log", type=str, default=None, help="Path to crawl_log.csv")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic data")
    args = parser.parse_args()

    if args.log is None and not args.demo:
        print("No --log provided. Running in demo mode (synthetic data).")
        args.demo = True

    run_pipeline(args.log, args.output, demo=args.demo)
