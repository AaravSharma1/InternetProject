"""
Compare semantic-prioritizer chunk sizes (50 / 150 / 300 chars of
surrounding text) on a synthetic labeled link-context dataset.

Metrics per chunk size:
  - ROC AUC of cosine score vs. topic centroid
  - Mean score for positives, mean score for negatives, separation gap
  - Mean per-URL scoring latency (ms)

Outputs (alongside this script):
  results.csv, results.json, auc_vs_chunk.png, separation_vs_chunk.png,
  latency_vs_chunk.png, score_distributions.png
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "3"))

from semantic_prioritizer import SemanticPrioritizer, URLContextExtractor  # noqa: E402

TOPIC_SEEDS = [
    "machine learning research papers",
    "deep learning neural networks",
    "artificial intelligence research",
]

# (anchor, surrounding_text, url, label)  label: 1 = relevant, 0 = irrelevant
# Surrounding text is long enough (>=300 chars) that 50/150/300 differ.
DATASET = [
    # ---- positives ----
    ("deep learning survey",
     "This survey covers the foundations of deep learning, including convolutional networks, recurrent architectures, and transformer models. It reviews recent results in image classification, speech recognition, and natural language processing, and discusses optimization, regularization, and generalization in modern deep neural networks trained on large-scale datasets.",
     "https://arxiv.org/cs/machine-learning/deep-learning-survey", 1),
    ("transformer architecture",
     "The transformer architecture introduced self-attention as a replacement for recurrence and convolution in sequence modeling. It has since become the dominant backbone for large language models, machine translation, and many vision tasks. We discuss multi-head attention, positional encoding, and scaling behavior of transformer models trained on web-scale corpora.",
     "https://papers.example.com/nlp/transformers-2023", 1),
    ("reinforcement learning tutorial",
     "Reinforcement learning studies how agents learn to act in an environment to maximize reward. This tutorial walks through Markov decision processes, value iteration, Q-learning, and modern policy gradient methods including PPO and SAC, with examples drawn from robotics, game playing, and recommendation systems research.",
     "https://ml.example.com/tutorials/rl/policy-gradient", 1),
    ("graph neural networks",
     "Graph neural networks generalize deep learning to graph-structured data. We review message passing, graph convolution, attention on graphs, and applications to molecular property prediction, knowledge graphs, and large-scale node classification benchmarks. Recent work explores scalability and expressivity of GNN architectures.",
     "https://research.example.com/gnn/node-classification", 1),
    ("convolutional neural network",
     "Convolutional neural networks remain the workhorse of computer vision. This article discusses CNN architectures from LeNet and AlexNet through ResNet and EfficientNet, training on ImageNet, transfer learning, and the relationship between CNNs and modern vision transformers used in image classification and segmentation tasks.",
     "https://vision.example.com/cnn/image-classification", 1),
    ("self-supervised learning",
     "Self-supervised representation learning trains models on pretext tasks derived from unlabeled data. Contrastive methods such as SimCLR and MoCo, masked image modeling, and language model pretraining have produced strong embeddings for downstream tasks. We review the theoretical and empirical landscape of SSL research.",
     "https://papers.example.com/ssl/contrastive", 1),
    ("large language model fine-tuning",
     "Large language models pretrained on web-scale text can be adapted to downstream tasks via supervised fine-tuning, instruction tuning, and reinforcement learning from human feedback. This article surveys parameter-efficient fine-tuning techniques such as LoRA and adapters and discusses evaluation of LLM reasoning ability.",
     "https://nlp.example.com/llm/fine-tuning", 1),
    ("diffusion model image generation",
     "Diffusion models have emerged as state-of-the-art generative models for images, audio, and video. We cover denoising diffusion probabilistic models, score-based generative models, and classifier-free guidance, plus practical training recipes used by Stable Diffusion and related text-to-image research systems.",
     "https://research.example.com/diffusion/generation", 1),
    ("attention mechanism",
     "Attention mechanisms allow neural networks to focus on relevant parts of their input. We trace attention from early seq2seq models through self-attention and the transformer, and discuss efficient attention variants used in long-context language models and high-resolution vision research.",
     "https://nlp.example.com/attention/overview", 1),
    ("optimization deep learning",
     "Optimization in deep learning involves stochastic gradient descent and its adaptive variants such as Adam and AdamW. We discuss learning rate schedules, weight decay, gradient clipping, and second-order methods, with empirical comparisons on standard image and language model training benchmarks.",
     "https://ml.example.com/optimization/sgd-adam", 1),

    # ---- negatives ----
    ("buy cheap sunglasses",
     "Save big on designer sunglasses with our summer clearance sale. Shop polarized lenses, aviators, and wayfarers from top brands. Free shipping on orders over fifty dollars and easy returns. Use code SUMMER for an extra ten percent off any pair of sunglasses or eyewear accessories in stock today.",
     "https://shop.example.com/accessories/sunglasses-sale", 0),
    ("chocolate cake recipe",
     "This rich chocolate cake recipe combines cocoa powder, dark chocolate, and buttermilk for a moist crumb. We walk through ingredient ratios, baking temperature and time, and a glossy ganache topping. Perfect for birthdays, anniversaries, or anytime you want a classic homemade chocolate dessert with simple pantry ingredients.",
     "https://food.example.com/desserts/chocolate-cake", 0),
    ("hotel booking deals",
     "Find the best hotel deals for your next vacation. Compare prices across hundreds of destinations, read verified guest reviews, and book rooms with free cancellation. From beach resorts to city center hotels, we help travelers save on accommodation worldwide for business trips, weekend getaways, and family holidays.",
     "https://travel.example.com/hotels/best-deals", 0),
    ("stock market earnings",
     "Quarterly earnings reports drove sharp moves across major indices today. Technology and energy sectors led gains while consumer staples lagged. Analysts revised guidance upward for several mega-cap firms after stronger-than-expected revenue. We summarize the key earnings beats and misses that shaped today's trading session.",
     "https://finance.example.com/stocks/earnings-report", 0),
    ("weather forecast weekly",
     "This week's forecast calls for mild temperatures and scattered afternoon showers across the region. Highs in the mid seventies with overnight lows near sixty. A weak cold front arriving Friday may bring stronger thunderstorms before drier weather returns over the weekend with sunny skies and lower humidity.",
     "https://weather.example.com/local/weekly-forecast", 0),
    ("football highlights week",
     "Catch up on this week's football highlights, including the top plays, biggest upsets, and standout individual performances. Our recap covers all the major games with video clips, box scores, and player of the game picks. Plus, a look ahead to next week's most-anticipated matchups across the league.",
     "https://sports.example.com/football/highlights-week3", 0),
    ("car insurance quotes",
     "Compare car insurance quotes from leading providers in minutes. Enter your zip code to see personalized rates based on your driving history, vehicle, and coverage preferences. Customers who switch save an average of several hundred dollars per year on auto insurance premiums for comparable liability and collision coverage.",
     "https://insurance.example.com/auto/quotes", 0),
    ("celebrity gossip news",
     "The latest celebrity gossip and entertainment news from Hollywood and beyond. Red carpet photos, breakups, reunions, and behind-the-scenes scoops from your favorite stars. Plus, exclusive interviews and rumor roundups covering film, television, and music industry happenings over the past week and weekend events.",
     "https://entertainment.example.com/gossip/this-week", 0),
    ("home gardening tips",
     "Our spring gardening guide covers soil preparation, choosing the right plants for your climate zone, and watering schedules. Learn how to start seeds indoors, transplant seedlings, and protect young plants from late frosts. We also cover common pests and organic methods to keep your vegetable garden healthy.",
     "https://home.example.com/gardening/spring-tips", 0),
    ("travel destinations europe",
     "Plan your next European getaway with our curated list of top destinations. From the canals of Venice to the beaches of the Algarve, we cover where to stay, what to eat, and the must-see attractions in each city. Tips on rail passes, budget accommodation, and the best time of year to visit.",
     "https://travel.example.com/europe/top-destinations", 0),

    # ---- HARD POSITIVES: weak anchor / weak URL, strong surrounding text.
    # A short chunk should miss the relevance; a longer chunk should catch it.
    ("read more",
     "Save up to seventy percent on home essentials this weekend only. Then later: researchers from a university lab published a new benchmark for evaluating large language models on multi-step reasoning, with results showing transformer-based architectures outperform recurrent baselines on math word problems and code generation tasks across several model scales.",
     "https://blog.example.com/posts/2024-04-12", 1),
    ("click here",
     "Welcome to our blog. Today's weather is mild. Now to the main topic: we present a study of self-supervised pretraining for vision transformers on medical imaging datasets, comparing masked autoencoder objectives with contrastive losses and reporting downstream classification AUC on three radiology benchmarks.",
     "https://blog.example.com/p/2024/04/13", 1),
    ("full article",
     "Yesterday's recipe roundup featured spring salads. After that section: a research note on sample-efficient reinforcement learning, demonstrating that model-based rollouts combined with prioritized experience replay reduce environment interactions by an order of magnitude on continuous control benchmarks compared with model-free baselines.",
     "https://blog.example.com/articles/april-14", 1),
    ("see post",
     "Quick housekeeping note about our newsletter schedule for the week. The featured study covers neural architecture search using gradient-based relaxation, finding cell structures that match hand-designed convolutional networks on image classification while using a fraction of the search compute reported in earlier work.",
     "https://blog.example.com/posts/april-15", 1),

    # ---- HARD NEGATIVES: ML-keyword-heavy anchor or URL, but surrounding text is off-topic.
    # A short chunk (anchor + URL only) will look relevant; a longer chunk should reveal it isn't.
    ("machine learning toy",
     "This plush toy is a soft, huggable bear made for toddlers. Machine washable on cold settings and tumble dry low. Features embroidered eyes for safety. A perfect gift for birthdays and baby showers. Also available in pink and blue. Free shipping on orders over twenty-five dollars in stock today.",
     "https://shop.example.com/toys/machine-learning-bear", 0),
    ("neural network costume",
     "Halloween is coming and our costume catalog is packed. This neural-network-themed costume includes a hooded jumpsuit and glow-in-the-dark accents. Sizes small through extra large. Returns accepted within thirty days unworn. Pair with our matching accessories for a complete look at this year's costume parties and events.",
     "https://shop.example.com/costumes/neural-network", 0),
    ("transformer party",
     "Throwing a transformer-themed birthday party? Our party pack includes plates, cups, napkins, and a banner. Suitable for ages five and up. Bundle with our balloon set for additional savings on your next themed celebration. Customer reviews give this kit four out of five stars for value and durability over time.",
     "https://shop.example.com/party/transformer-pack", 0),
    ("deep learning gym",
     "Our fitness studio offers strength, mobility, and high-intensity training classes seven days a week. New member specials include a free trial week and discounted personal training packages. Locker rooms, showers, and on-site parking included. Drop in for a tour and meet our certified coaches anytime this month.",
     "https://gym.example.com/membership", 0),
]


def build_context(extractor: URLContextExtractor, anchor: str, surrounding: str, url: str, chunk: int) -> str:
    # Replicate URLContextExtractor.extract but with chunk as the surrounding-text cap
    url_tokens = extractor._tokenize_url_path(url)
    parts = [anchor.strip(), surrounding.strip()[:chunk], url_tokens]
    return " ".join(p for p in parts if p)


def run_for_chunk(prioritizer: SemanticPrioritizer, extractor: URLContextExtractor,
                  centroid: np.ndarray, chunk: int, latency_trials: int = 5) -> dict:
    contexts = [build_context(extractor, a, s, u, chunk) for a, s, u, _ in DATASET]
    labels = [lab for *_, lab in DATASET]

    scores = prioritizer.score_batch(contexts, centroid)
    auc = roc_auc_score(labels, scores)

    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]

    # warm up
    for _ in range(3):
        prioritizer.score_batch(contexts, centroid)

    # latency: time to score the whole batch, divided by batch size, averaged
    per_url_ms = []
    for _ in range(latency_trials):
        t0 = time.perf_counter()
        prioritizer.score_batch(contexts, centroid)
        per_url_ms.append((time.perf_counter() - t0) * 1000.0 / len(contexts))

    avg_chars = statistics.mean(len(c) for c in contexts)

    return {
        "chunk": chunk,
        "auc": float(auc),
        "mean_pos": float(np.mean(pos)),
        "mean_neg": float(np.mean(neg)),
        "separation": float(np.mean(pos) - np.mean(neg)),
        "latency_ms_per_url": float(statistics.mean(per_url_ms)),
        "latency_stdev_ms": float(statistics.stdev(per_url_ms)) if len(per_url_ms) > 1 else 0.0,
        "avg_context_chars": avg_chars,
        "scores": scores,
        "labels": labels,
    }


def plot_bar(values, ylabel, title, out, color="steelblue", hline=None, hline_label=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [str(v["chunk"]) for v in values]
    ys = [v["value"] for v in values]
    bars = ax.bar(xs, ys, color=color, alpha=0.85, edgecolor="white")
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", label=hline_label)
        ax.legend()
    ax.set_xlabel("Surrounding-text chunk size (chars)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


def plot_distributions(results, out):
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        pos = [s for s, l in zip(r["scores"], r["labels"]) if l == 1]
        neg = [s for s, l in zip(r["scores"], r["labels"]) if l == 0]
        ax.hist(neg, bins=10, color="lightcoral", alpha=0.7, label="irrelevant", edgecolor="white")
        ax.hist(pos, bins=10, color="steelblue", alpha=0.7, label="relevant", edgecolor="white")
        ax.set_title(f"chunk = {r['chunk']} chars\nAUC = {r['auc']:.3f}")
        ax.set_xlabel("cosine score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


def main():
    out_dir = Path(__file__).resolve().parent
    chunk_sizes = [50, 150, 300]

    print("Loading SemanticPrioritizer...")
    prioritizer = SemanticPrioritizer()
    extractor = URLContextExtractor()
    centroid = prioritizer.init_centroid(TOPIC_SEEDS)

    print(f"Dataset: {len(DATASET)} contexts ({sum(l for *_, l in DATASET)} positive)")
    print(f"Topic seeds: {TOPIC_SEEDS}")

    results = []
    for c in chunk_sizes:
        print(f"\n--- chunk = {c} chars ---")
        r = run_for_chunk(prioritizer, extractor, centroid, c)
        print(f"  avg context length: {r['avg_context_chars']:.0f} chars")
        print(f"  AUC: {r['auc']:.4f}")
        print(f"  mean pos / neg: {r['mean_pos']:.4f} / {r['mean_neg']:.4f}  (gap {r['separation']:.4f})")
        print(f"  latency / URL: {r['latency_ms_per_url']:.3f} ms (+/- {r['latency_stdev_ms']:.3f})")
        results.append(r)

    # CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w") as fh:
        fh.write("chunk_chars,avg_context_chars,auc,mean_relevant,mean_irrelevant,separation_gap,latency_ms_per_url,latency_stdev_ms\n")
        for r in results:
            fh.write(f"{r['chunk']},{r['avg_context_chars']:.1f},{r['auc']:.4f},"
                     f"{r['mean_pos']:.4f},{r['mean_neg']:.4f},{r['separation']:.4f},"
                     f"{r['latency_ms_per_url']:.4f},{r['latency_stdev_ms']:.4f}\n")
    print(f"\nSaved {csv_path}")

    # JSON (without raw scores arrays for cleanliness)
    summary = [{k: v for k, v in r.items() if k not in ("scores", "labels")} for r in results]
    with open(out_dir / "results.json", "w") as fh:
        json.dump({"topic_seeds": TOPIC_SEEDS, "n_examples": len(DATASET), "results": summary}, fh, indent=2)
    print(f"Saved {out_dir / 'results.json'}")

    # Plots
    plot_bar([{"chunk": r["chunk"], "value": r["auc"]} for r in results],
             "ROC AUC (relevant vs irrelevant)",
             "Classification quality vs chunk size",
             out_dir / "auc_vs_chunk.png", hline=0.5, hline_label="chance")

    plot_bar([{"chunk": r["chunk"], "value": r["separation"]} for r in results],
             "mean(relevant) − mean(irrelevant)",
             "Score separation gap vs chunk size",
             out_dir / "separation_vs_chunk.png", color="seagreen")

    plot_bar([{"chunk": r["chunk"], "value": r["latency_ms_per_url"]} for r in results],
             "Latency per URL (ms)",
             "Scoring latency vs chunk size",
             out_dir / "latency_vs_chunk.png", color="darkorange",
             hline=10.0, hline_label="10 ms target")

    plot_distributions(results, out_dir / "score_distributions.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
