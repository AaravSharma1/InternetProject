import argparse
import statistics
import time

import matplotlib.pyplot as plt

from semantic_prioritizer import SemanticPrioritizer

SAMPLE_CONTEXTS = [
    "deep learning survey paper /cs/machine-learning/deep-learning",
    "click here for more information /news/sports/latest-scores",
    "transformer architecture attention NLP research /papers/transformers-2023",
    "buy cheap sunglasses discount sale /shop/accessories/sunglasses",
    "reinforcement learning policy gradient tutorial /tutorials/rl/policy",
    "recipe chocolate cake baking ingredients /food/desserts/chocolate-cake",
    "graph neural network node classification benchmark /research/gnn/node",
    "hotel booking deals travel vacation /travel/hotels/best-deals",
    "convolutional neural network image classification vision /vision/cnn",
    "stock market analysis financial earnings report /finance/stocks/earnings",
    "self-supervised learning contrastive representation /papers/ssl/contrastive",
    "weather forecast temperature humidity /weather/local/weekly",
    "large language model pre-training fine-tuning /nlp/llm/fine-tuning",
    "sports highlights football basketball /sports/highlights/week3",
    "diffusion model image generation stable /research/diffusion/generation",
]


def _make_contexts(n):
    return [SAMPLE_CONTEXTS[i % len(SAMPLE_CONTEXTS)] for i in range(n)]


def benchmark_single_url(prioritizer, n_trials=100):
    centroid = prioritizer.init_centroid(["machine learning research papers"])
    ctx = SAMPLE_CONTEXTS[0]

    # warm up
    for _ in range(5):
        prioritizer.score(ctx, centroid)

    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        prioritizer.score(ctx, centroid)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "stdev_ms": statistics.stdev(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * n_trials)],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "passes_10ms": statistics.mean(latencies) < 10.0,
        "raw": latencies,
    }


def benchmark_batch_sizes(prioritizer, batch_sizes=(1, 10, 50, 100), n_trials=20):
    centroid = prioritizer.init_centroid(["machine learning research papers"])
    results = {}

    for bs in batch_sizes:
        contexts = _make_contexts(bs)

        # warm up
        for _ in range(3):
            prioritizer.score_batch(contexts, centroid)

        total_ms = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            prioritizer.score_batch(contexts, centroid)
            total_ms.append((time.perf_counter() - t0) * 1000)

        mean_total = statistics.mean(total_ms)
        results[bs] = {
            "mean_total_ms": mean_total,
            "median_total_ms": statistics.median(total_ms),
            "stdev_ms": statistics.stdev(total_ms) if len(total_ms) > 1 else 0.0,
            "mean_per_url_ms": mean_total / bs,
        }

    return results


def plot_single(single, output="benchmark_single.png"):
    latencies = single["raw"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(latencies, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(single["mean_ms"], color="navy", linestyle="--", label=f"mean {single['mean_ms']:.1f} ms")
    ax.axvline(10.0, color="red", linestyle=":", label="10 ms target")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Single-URL Scoring Latency Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"  Saved: {output}")
    plt.close()


def plot_batch(batch, output="benchmark_batch.png"):
    batch_sizes = sorted(batch.keys())
    per_url = [batch[bs]["mean_per_url_ms"] for bs in batch_sizes]
    total = [batch[bs]["mean_total_ms"] for bs in batch_sizes]
    errs = [batch[bs]["stdev_ms"] for bs in batch_sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(batch_sizes, per_url, marker="o", color="steelblue", linewidth=2)
    ax1.axhline(10.0, color="red", linestyle="--", label="10 ms target")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Mean latency per URL (ms)")
    ax1.set_title("Per-URL Latency vs. Batch Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar([str(bs) for bs in batch_sizes], total, yerr=errs, color="steelblue", alpha=0.8, capsize=5)
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Total latency (ms)")
    ax2.set_title("Total Batch Latency vs. Batch Size")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"  Saved: {output}")
    plt.close()


def print_summary(single, batch):
    print("\n" + "=" * 62)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 62)

    print(f"\nSingle-URL scoring (n={len(single['raw'])} trials):")
    print(f"  Mean      : {single['mean_ms']:7.2f} ms")
    print(f"  Median    : {single['median_ms']:7.2f} ms")
    print(f"  Std dev   : {single['stdev_ms']:7.2f} ms")
    print(f"  P95       : {single['p95_ms']:7.2f} ms")
    print(f"  Min / Max : {single['min_ms']:.2f} / {single['max_ms']:.2f} ms")
    print(f"  < 10 ms   : {'PASS' if single['passes_10ms'] else 'FAIL'}")

    print(f"\nBatch scoring:")
    print(f"  {'Batch':>8}  {'Total (ms)':>12}  {'Per-URL (ms)':>14}  {'Std (ms)':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*14}  {'-'*10}")
    for bs in sorted(batch.keys()):
        r = batch[bs]
        print(f"  {bs:>8}  {r['mean_total_ms']:>12.2f}  {r['mean_per_url_ms']:>14.4f}  {r['stdev_ms']:>10.2f}")

    if 1 in batch and len(batch) > 1:
        base = batch[1]["mean_per_url_ms"]
        print(f"\nSpeedup relative to batch=1 (per-URL):")
        for bs in sorted(batch.keys()):
            speedup = base / batch[bs]["mean_per_url_ms"] if batch[bs]["mean_per_url_ms"] > 0 else float("inf")
            print(f"  batch={bs:>4}: {speedup:.2f}x")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--batch-trials", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    prioritizer = SemanticPrioritizer()

    print(f"Benchmarking single-URL latency ({args.trials} trials)...")
    single = benchmark_single_url(prioritizer, n_trials=args.trials)

    print(f"Benchmarking batch sizes {args.batch_sizes} ({args.batch_trials} trials each)...")
    batch = benchmark_batch_sizes(prioritizer, batch_sizes=tuple(args.batch_sizes), n_trials=args.batch_trials)

    print_summary(single, batch)

    if not args.no_plot:
        plot_single(single)
        plot_batch(batch)


if __name__ == "__main__":
    main()
