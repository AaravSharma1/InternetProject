"""
eval_pipeline.py — Evaluation Pipeline for Semantic Web Crawler
Person 4: Evaluation, Visualization & Demo

────────────────────────────────────────────────────────────────────────────────
CSV SCHEMA  (produced by Person 1 — crawler/metrics.py)
────────────────────────────────────────────────────────────────────────────────
Column                  Notes
----------------------  --------------------------------------------------------
timestamp               Unix epoch float  →  we parse as datetime
url                     Fetched URL
bytes_downloaded        Raw bytes received (0 on error)
fetch_latency_ms        Wall-clock fetch time in ms
status_code             HTTP status (0 on connection failure)
is_relevant             Boolean set by caller's relevance_fn
cumulative_pages        Running total of fetched pages
cumulative_relevant     Running total of relevant pages
error                   Error message or empty string

ADDED BY US (multi-run experiment harness — see load_mode_csvs()):
  mode      "bfs" | "link_priority" | "semantic"   (injected before analysis)
  node_id   "node_1" … "node_N"                    (optional; defaults to "node_1")
  run       integer run number 1–3                 (optional; for mean±std)

Columns absent from a plain Person-1 CSV get safe defaults injected
automatically so the script works with zero changes to their code.

────────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────────
# Demo on synthetic data (no real CSV needed):
    python eval_pipeline.py --demo

# Single CSV that already has a 'mode' column:
    python eval_pipeline.py --log crawl_metrics.csv --output results/

# Three separate CSVs, one per mode (most common during integration week):
    python eval_pipeline.py \
        --bfs  runs/bfs_crawl_metrics.csv \
        --link runs/link_crawl_metrics.csv \
        --sem  runs/semantic_crawl_metrics.csv \
        --output results/

# Multiple runs per mode for mean±std (comma-separated paths):
    python eval_pipeline.py \
        --bfs  bfs_r1.csv,bfs_r2.csv,bfs_r3.csv \
        --link link_r1.csv,link_r2.csv,link_r3.csv \
        --sem  sem_r1.csv,sem_r2.csv,sem_r3.csv \
        --output results/

────────────────────────────────────────────────────────────────────────────────
DEPENDENCIES
────────────────────────────────────────────────────────────────────────────────
    pip install pandas matplotlib datasketch numpy
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datasketch import MinHash, MinHashLSH

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Constants / schema contract
# ─────────────────────────────────────────────────────────────────────────────

# Columns Person 1 guarantees (crawler/metrics.py FIELDNAMES)
P1_REQUIRED_COLS = {
    "timestamp",
    "url",
    "bytes_downloaded",      # NOTE: NOT "bytes" — matches Person 1 exactly
    "fetch_latency_ms",      # NOTE: NOT "fetch_ms"
    "status_code",
    "is_relevant",
    "cumulative_pages",
    "cumulative_relevant",
}

# Columns our harness adds (absent from plain P1 CSV → defaults injected)
RELEVANCE_SCORE_COL = "relevance_score"   # float 0-1 from Person 3's scorer
PAGE_TEXT_COL       = "page_text"         # full page text for MinHash
NODE_ID_COL         = "node_id"
MODE_COL            = "mode"
RUN_COL             = "run"

MODES = ["bfs", "link_priority", "semantic"]
MODE_LABELS = {
    "bfs":           "BFS baseline",
    "link_priority": "Link-priority",
    "semantic":      "Semantic (ours)",
}
MODE_COLORS = {
    "bfs":           "#888780",
    "link_priority": "#378ADD",
    "semantic":      "#1D9E75",
}
MODE_DASHES = {
    "bfs":           (6, 2),
    "link_priority": (3, 1),
    "semantic":      None,
}

JACCARD_THRESHOLD = 0.8
MINHASH_NUM_PERM  = 128
SHINGLE_SIZE      = 5   # 5-word shingles


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loading & normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _load_single_csv(path: str) -> pd.DataFrame:
    """
    Load one of Person 1's crawl_metrics.csv files and validate column names.
    Injects safe defaults for the extra columns our harness expects.
    """
    df = pd.read_csv(path)

    missing = P1_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"\n{path}: missing required columns: {missing}"
            f"\n  Expected (from crawler/metrics.py): {sorted(P1_REQUIRED_COLS)}"
            f"\n  Found:   {sorted(df.columns.tolist())}"
        )

    # timestamp: Person 1 stores Unix epoch floats
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    df["is_relevant"]      = df["is_relevant"].astype(bool)
    df["bytes_downloaded"] = pd.to_numeric(df["bytes_downloaded"], errors="coerce").fillna(0)
    df["fetch_latency_ms"] = pd.to_numeric(df["fetch_latency_ms"], errors="coerce").fillna(0)
    df["status_code"]      = pd.to_numeric(df["status_code"],      errors="coerce").fillna(0).astype(int)
    df["cumulative_pages"]    = pd.to_numeric(df["cumulative_pages"],    errors="coerce").fillna(0)
    df["cumulative_relevant"] = pd.to_numeric(df["cumulative_relevant"], errors="coerce").fillna(0)

    # Inject defaults for columns absent from plain P1 CSVs
    if MODE_COL not in df.columns:
        df[MODE_COL] = "unknown"
    if NODE_ID_COL not in df.columns:
        df[NODE_ID_COL] = "node_1"
    if RUN_COL not in df.columns:
        df[RUN_COL] = 1
    if RELEVANCE_SCORE_COL not in df.columns:
        # Derive a pseudo-score so plots still render
        df[RELEVANCE_SCORE_COL] = df["is_relevant"].astype(float)

    df[MODE_COL] = df[MODE_COL].str.lower().str.strip()
    return df.sort_values("timestamp").reset_index(drop=True)


def load_log(path: str) -> pd.DataFrame:
    """Load a single CSV (already has a 'mode' column, or defaults to 'unknown')."""
    return _load_single_csv(path)


def load_mode_csvs(
    bfs_paths:  list | None = None,
    link_paths: list | None = None,
    sem_paths:  list | None = None,
) -> pd.DataFrame:
    """
    Load separate per-mode CSV files and tag each with mode + run number.
    This is the normal integration-week workflow:
      Person 1 runs the crawler three times per mode → 9 CSVs total.
    """
    frames = []
    for mode_key, paths in [
        ("bfs",           bfs_paths  or []),
        ("link_priority", link_paths or []),
        ("semantic",      sem_paths  or []),
    ]:
        for run_num, p in enumerate(paths, start=1):
            df = _load_single_csv(p)
            df[MODE_COL] = mode_key      # overwrite whatever was in the file
            df[RUN_COL]  = run_num
            frames.append(df)
            print(f"  Loaded {len(df):>6,} rows  mode={mode_key:14s}  run={run_num}  ({p})")

    if not frames:
        raise ValueError("No CSV paths provided.")

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values([MODE_COL, RUN_COL, "timestamp"]).reset_index(drop=True)


def make_synthetic_log(n_pages: int = 3000, runs: int = 3) -> pd.DataFrame:
    """
    Synthetic data matching Person 1's exact CSV schema.
    Used for standalone testing — no crawl required.
    """
    rng = np.random.default_rng(42)
    rows = []
    t0 = 1_700_000_000.0  # Unix epoch base

    for run in range(1, runs + 1):
        for mode, rel_rate in [("bfs", 0.18), ("link_priority", 0.32), ("semantic", 0.61)]:
            cum_pages = 0
            cum_rel   = 0
            for i in range(n_pages):
                # Semantic front-loads relevant pages
                if mode == "semantic":
                    local_rate = rel_rate * (1.4 - 0.8 * i / n_pages)
                else:
                    local_rate = rel_rate
                is_rel = bool(rng.random() < max(0.0, min(1.0, local_rate)))
                cum_pages += 1
                cum_rel   += int(is_rel)
                rows.append({
                    # Person 1 columns (exact names)
                    "timestamp":          t0 + i * 0.3 + run * 5000,
                    "url":                f"https://example.com/{mode}/{run}/{i}",
                    "bytes_downloaded":   int(rng.integers(5_000, 200_000)),
                    "fetch_latency_ms":   float(rng.integers(50, 800)),
                    "status_code":        200,
                    "is_relevant":        is_rel,
                    "cumulative_pages":   cum_pages,
                    "cumulative_relevant": cum_rel,
                    "error":              "",
                    # Our harness columns
                    MODE_COL:             mode,
                    NODE_ID_COL:          f"node_{rng.integers(1, 5)}",
                    RUN_COL:              run,
                    RELEVANCE_SCORE_COL:  float(rng.uniform(0.6, 0.99) if is_rel
                                               else rng.uniform(0.01, 0.45)),
                    PAGE_TEXT_COL:        f"sample text page {i} mode {mode} run {run} " * 20,
                })

    df = pd.DataFrame(rows)
    df["timestamp"]   = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["is_relevant"] = df["is_relevant"].astype(bool)
    return df.sort_values([MODE_COL, RUN_COL, "timestamp"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MinHash redundancy detection
# ─────────────────────────────────────────────────────────────────────────────

def _shinglize(text: str, k: int = SHINGLE_SIZE) -> set:
    words = text.lower().split()
    if len(words) < k:
        return {text.lower()}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _make_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    for shingle in _shinglize(str(text)):
        m.update(shingle.encode("utf-8"))
    return m


def compute_redundancy_rate(df: pd.DataFrame) -> dict:
    """
    Near-duplicate rate per mode via MinHash LSH.

    Text source priority:
      1. page_text column  — actual HTML text (best accuracy; optional column)
      2. url column        — URL fingerprint fallback (always present in P1 CSV)

    Returns {mode: rate_0_to_1}
    """
    has_text = PAGE_TEXT_COL in df.columns
    if not has_text:
        print("  [warn] 'page_text' column absent — using URL as text proxy for MinHash")

    results: dict = {}
    for mode, group in df.groupby(MODE_COL):
        lsh        = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=MINHASH_NUM_PERM)
        duplicates = 0
        total      = 0
        for idx, row in group.iterrows():
            text = str(row[PAGE_TEXT_COL]) if has_text else str(row["url"])
            mh   = _make_minhash(text)
            key  = f"r_{idx}"
            try:
                if lsh.query(mh):
                    duplicates += 1
                else:
                    lsh.insert(key, mh)
            except Exception:
                try:
                    lsh.insert(key, mh)
                except Exception:
                    pass
            total += 1
        rate = duplicates / total if total > 0 else 0.0
        results[mode] = rate
        print(f"  [{mode:14s}]  redundancy={rate:.1%}  ({duplicates:,}/{total:,})")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core metrics  (all column references use P1's actual names)
# ─────────────────────────────────────────────────────────────────────────────

def harvest_rate(df: pd.DataFrame) -> dict:
    """Fraction of crawled pages flagged is_relevant, per mode."""
    return df.groupby(MODE_COL)["is_relevant"].mean().to_dict()


def relevance_at_depth(df: pd.DataFrame, sample_points: int = 100) -> dict:
    """
    Cumulative harvest rate as pages accumulate, in discovery order.
    Prefers Person 1's cumulative_relevant / cumulative_pages columns
    (already correct even for multi-node runs), falling back to recomputing.
    Returns {mode: (x_array, y_array)}
    """
    result = {}
    for mode, group in df.groupby(MODE_COL):
        # Use run=1 for the curve; multi-run variance shown in mean±std table
        run_ids = sorted(group[RUN_COL].unique())
        first   = group[group[RUN_COL] == run_ids[0]].reset_index(drop=True)
        n  = len(first)
        xi = np.linspace(0, n - 1, min(sample_points, n), dtype=int)
        cp = first["cumulative_pages"].iloc[xi].values
        cr = first["cumulative_relevant"].iloc[xi].values
        yi = np.where(cp > 0, cr / cp, 0.0)
        result[mode] = (xi, yi)
    return result


def bandwidth_efficiency(df: pd.DataFrame) -> dict:
    """
    Cumulative MB downloaded per relevant page discovered.
    Uses bytes_downloaded (P1 column name).
    """
    result = {}
    for mode, group in df.groupby(MODE_COL):
        run_ids   = sorted(group[RUN_COL].unique())
        first     = group[group[RUN_COL] == run_ids[0]].reset_index(drop=True)
        cum_bytes = first["bytes_downloaded"].cumsum() / 1e6
        cum_rel   = first["is_relevant"].cumsum().replace(0, np.nan)
        result[mode] = (cum_bytes / cum_rel).values
    return result


def throughput_per_node(df: pd.DataFrame) -> dict:
    """
    Pages/sec per node.
    Uses fetch_latency_ms (P1 column name) and timestamp to compute span.
    """
    result = {}
    for mode, group in df.groupby(MODE_COL):
        n_nodes = group[NODE_ID_COL].nunique()
        t_span  = (group["timestamp"].max() - group["timestamp"].min()).total_seconds()
        if t_span <= 0:
            # Fallback: sum of all fetch times / parallelism
            t_span = group["fetch_latency_ms"].sum() / 1000 / max(n_nodes, 1)
        result[mode] = round(len(group) / max(t_span, 1) / max(n_nodes, 1), 2)
    return result


def cross_node_bytes(df: pd.DataFrame) -> dict:
    """
    Estimated inter-node URL forwarding overhead.
    Formula: ~50% of discovered URLs land on the wrong shard, each costs 100 bytes.
    For exact numbers, Person 2 should expose a Redis counter per node.
    """
    URL_BYTES = 100
    result = {}
    for mode, group in df.groupby(MODE_COL):
        n_nodes   = group[NODE_ID_COL].nunique()
        forwarded = len(group) * (1 - 1 / max(n_nodes, 1)) * 0.5
        result[mode] = int(forwarded * URL_BYTES)
    return result


def mean_std_across_runs(df: pd.DataFrame, metric_fn) -> pd.DataFrame:
    """Compute mean ± std of a scalar-per-mode metric across all run numbers."""
    if RUN_COL not in df.columns or df[RUN_COL].nunique() < 2:
        return pd.DataFrame()
    records = [metric_fn(df[df[RUN_COL] == r]) for r in sorted(df[RUN_COL].unique())]
    all_keys = set().union(*records)
    rows = {}
    for k in all_keys:
        vals = [rec[k] for rec in records if k in rec]
        rows[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#D3D1C7")
    ax.tick_params(colors="#5F5E5A", labelsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#2C2C2A", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#5F5E5A")
    ax.set_ylabel(ylabel, fontsize=9, color="#5F5E5A")
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.5, linestyle="--")


def plot_relevance_at_depth(rad: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for mode, (x, y) in rad.items():
        dash = MODE_DASHES[mode]
        ls   = (0, dash) if dash else "-"
        ax.plot(x, y * 100, color=MODE_COLORS[mode], linestyle=ls, linewidth=1.8,
                label=MODE_LABELS[mode],
                marker="o" if mode == "semantic" else None,
                markevery=10, markersize=3)
    _style_ax(ax, "Relevance at depth", "Pages crawled", "Cumulative harvest rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_bandwidth_efficiency(bwe: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for mode, vals in bwe.items():
        dash = MODE_DASHES[mode]
        ls   = (0, dash) if dash else "-"
        ax.plot(np.arange(len(vals)), vals,
                color=MODE_COLORS[mode], linestyle=ls, linewidth=1.8,
                label=MODE_LABELS[mode])
    _style_ax(ax, "Bandwidth efficiency", "Pages crawled", "MB per relevant page")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_throughput_bar(tpn: dict, output_path: str):
    modes = [m for m in MODES if m in tpn]
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    bars = ax.bar([MODE_LABELS[m] for m in modes], [tpn[m] for m in modes],
                  color=[MODE_COLORS[m] for m in modes], width=0.5, zorder=3)
    for bar, v in zip(bars, [tpn[m] for m in modes]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#2C2C2A")
    _style_ax(ax, "Throughput per node", "", "Pages / sec / node")
    ax.set_xticklabels([MODE_LABELS[m] for m in modes], fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_redundancy_bar(redundancy: dict, output_path: str):
    modes = [m for m in MODES if m in redundancy]
    vals  = [redundancy[m] * 100 for m in modes]
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    bars = ax.bar([MODE_LABELS[m] for m in modes], vals,
                  color=[MODE_COLORS[m] for m in modes], width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color="#2C2C2A")
    _style_ax(ax, "Near-duplicate rate by mode", "", "Near-duplicate pages (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xticklabels([MODE_LABELS[m] for m in modes], fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_table(df: pd.DataFrame, redundancy: dict) -> pd.DataFrame:
    hr  = harvest_rate(df)
    tpn = throughput_per_node(df)
    cnb = cross_node_bytes(df)
    rows = []
    for mode in MODES:
        if mode not in df[MODE_COL].values:
            continue
        sub     = df[df[MODE_COL] == mode]
        rel     = sub["is_relevant"].sum()
        total_mb = sub["bytes_downloaded"].sum() / 1e6
        rows.append({
            "Mode":                     MODE_LABELS.get(mode, mode),
            "Harvest rate (%)":         f"{hr.get(mode, 0) * 100:.1f}",
            "MB / relevant page":       f"{total_mb / rel:.2f}" if rel > 0 else "∞",
            "Redundancy (%)":           f"{redundancy.get(mode, 0) * 100:.1f}",
            "Throughput (pg/s/node)":   f"{tpn.get(mode, 0):.2f}",
            "Cross-node overhead (KB)": f"{cnb.get(mode, 0) / 1e3:.1f}",
            "Pages crawled":            len(sub),
        })
    return pd.DataFrame(rows).set_index("Mode")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _path_list(s: str | None) -> list:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def run_pipeline(args):
    out = args.output
    os.makedirs(out, exist_ok=True)
    print("\n=== Semantic Crawler — Evaluation Pipeline ===\n")

    if args.demo:
        print("Running on synthetic demo data (no CSV required).")
        df = make_synthetic_log(n_pages=3000, runs=3)
    elif args.log:
        print(f"Loading single log: {args.log}")
        df = load_log(args.log)
    elif any([args.bfs, args.link, args.sem]):
        print("Loading per-mode CSVs:")
        df = load_mode_csvs(
            bfs_paths  = _path_list(args.bfs),
            link_paths = _path_list(args.link),
            sem_paths  = _path_list(args.sem),
        )
    else:
        print("No input specified — running demo mode.")
        df = make_synthetic_log(n_pages=3000, runs=3)

    print(f"\nLoaded {len(df):,} rows | modes: {sorted(df[MODE_COL].unique())}")
    print(df.groupby(MODE_COL).agg(
        pages      = ("is_relevant", "count"),
        relevant   = ("is_relevant", "sum"),
        total_mb   = ("bytes_downloaded", lambda x: round(x.sum() / 1e6, 1)),
    ).to_string(), "\n")

    print("[1/5] Computing MinHash redundancy rates...")
    redundancy = compute_redundancy_rate(df)

    print("\n[2/5] Computing harvest rates, throughput, bandwidth...")
    rad = relevance_at_depth(df)
    bwe = bandwidth_efficiency(df)
    tpn = throughput_per_node(df)

    print("\n[3/5] Building summary table...")
    summary    = build_summary_table(df, redundancy)
    table_path = os.path.join(out, "summary_table.csv")
    summary.to_csv(table_path)
    print(f"\n{'─'*60}\n{summary.to_string()}\n{'─'*60}")
    print(f"  Saved → {table_path}\n")

    print("[4/5] Generating figures...")
    plot_relevance_at_depth(rad, os.path.join(out, "relevance_at_depth.png"))
    plot_bandwidth_efficiency(bwe, os.path.join(out, "bandwidth_efficiency.png"))
    plot_throughput_bar(tpn,       os.path.join(out, "throughput_bar.png"))
    plot_redundancy_bar(redundancy, os.path.join(out, "redundancy_bar.png"))

    if RUN_COL in df.columns and df[RUN_COL].nunique() >= 2:
        print("\n[5/5] Computing mean ± std across runs...")
        stats = mean_std_across_runs(df, harvest_rate)
        stats_path = os.path.join(out, "harvest_rate_stats.csv")
        stats.to_csv(stats_path)
        print(f"  Harvest-rate mean±std:\n{stats.to_string()}")
        print(f"  Saved → {stats_path}")
    else:
        print("[5/5] Skipped run stats (fewer than 2 runs detected).")

    print(f"\n=== Done — all outputs in {out}/ ===\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Semantic Crawler — Evaluation Pipeline (Person 4)")
    p.add_argument("--log",    type=str, default=None,
                   help="Single crawl_metrics.csv (needs a 'mode' column already)")
    p.add_argument("--bfs",    type=str, default=None,
                   help="BFS CSV path(s), comma-separated for multiple runs")
    p.add_argument("--link",   type=str, default=None,
                   help="Link-priority CSV path(s), comma-separated")
    p.add_argument("--sem",    type=str, default=None,
                   help="Semantic CSV path(s), comma-separated")
    p.add_argument("--output", type=str, default="results",
                   help="Output directory (default: results/)")
    p.add_argument("--demo",   action="store_true",
                   help="Run on synthetic data — no CSV required")
    args = p.parse_args()
    run_pipeline(args)
