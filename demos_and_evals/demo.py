import argparse
import random

import numpy as np

from mock_frontier import MockFrontier
from relevance_classifier import RelevanceClassifier
from semantic_prioritizer import SemanticPrioritizer, URLContextExtractor


TOPIC_SEEDS = [
    "machine learning research papers",
    "deep learning neural networks survey",
    "artificial intelligence academic publications",
]


def walkthrough_score(prioritizer):
    _header("WALKTHROUGH 1: score() end-to-end")

    centroid = prioritizer.init_centroid(TOPIC_SEEDS)
    print(f"  Centroid initialized from {len(TOPIC_SEEDS)} seed descriptions.")
    print(f"  Shape: {centroid.shape},  norm: {np.linalg.norm(centroid):.6f}\n")

    extractor = URLContextExtractor()

    cases = [
        (
            "BEST CASE  (~0.85 expected)",
            "deep learning survey paper NeurIPS 2023",
            "We present a comprehensive survey of deep learning methods across computer vision, NLP, and speech recognition. Our analysis covers over 500 recent publications and identifies key research trends.",
            "https://arxiv.org/abs/cs/deep-learning-survey-neurips2023",
        ),
        (
            "AVERAGE    (~0.45 expected)",
            "click here",
            "For more information about our research group, upcoming seminars, and faculty profiles, please click the link below to visit the department homepage.",
            "https://cs.university.edu/faculty/profile/homepage",
        ),
        (
            "WORST CASE (~0.10 expected)",
            "Buy now, 50% off!",
            "Limited time offer! Up to 70% off on sunglasses, shoes, and accessories. Free shipping on all orders over $50. Use code SALE50 at checkout.",
            "https://shop.deals.com/sale/accessories/sunglasses-2024",
        ),
    ]

    for label, anchor, surrounding, url in cases:
        context = extractor.extract(anchor, surrounding, url)
        score = prioritizer.score(context, centroid)
        print(f"  [{label}]")
        print(f"    anchor  : {anchor!r}")
        print(f"    url     : {url}")
        print(f"    context : {context[:90]}...")
        print(f"    score   : {score:.4f}")
        print()


def _relevant_pages(n):
    templates = [
        "This paper presents a novel deep learning approach for {task} using transformer architectures achieving state-of-the-art performance.",
        "We propose a new {task} algorithm based on gradient descent and self-supervised pre-training objectives.",
        "Recent advances in {task} have enabled breakthroughs in computer vision, NLP, and multi-modal learning.",
        "Our {task} framework is evaluated on standard benchmarks including ImageNet, GLUE, and SQuAD.",
        "Empirical study of {task} scaling laws shows consistent improvements with model size and compute.",
    ]
    tasks = [
        "deep learning", "reinforcement learning", "graph neural networks",
        "contrastive learning", "diffusion models", "large language models",
        "neural architecture search", "meta-learning", "few-shot learning",
    ]
    rng = random.Random(42)
    return [rng.choice(templates).format(task=rng.choice(tasks)) for _ in range(n)]


def walkthrough_centroid_drift(prioritizer, save_plot=True, output="centroid_drift.png"):
    _header("WALKTHROUGH 2: Centroid drift")

    centroid = prioritizer.init_centroid(TOPIC_SEEDS)
    snapshots = {"init": centroid.copy()}

    pages = _relevant_pages(1000)
    checkpoints = {100, 500, 1000}

    for i, page_text in enumerate(pages, start=1):
        centroid = prioritizer.update_centroid(centroid, page_text, score=None)
        if i in checkpoints:
            snapshots[i] = centroid.copy()

    init = snapshots["init"]
    print("  Cosine similarity between initial centroid and each checkpoint:")
    for step, snap in snapshots.items():
        if step == "init":
            continue
        sim = float(np.dot(init / (np.linalg.norm(init) + 1e-10), snap / (np.linalg.norm(snap) + 1e-10)))
        print(f"    init -> {step:>4} pages : {sim:.4f}")
    print()

    if not save_plot:
        return

    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        sample_texts = _relevant_pages(30)
        sample_embs = prioritizer.embed(sample_texts)

        snap_keys = list(snapshots.keys())
        snap_vectors = np.stack([snapshots[k] for k in snap_keys])
        all_vectors = np.vstack([snap_vectors, sample_embs])

        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(all_vectors)
        snap_2d = reduced[:len(snap_keys)]
        samp_2d = reduced[len(snap_keys):]

        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(snap_keys)))
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(samp_2d[:, 0], samp_2d[:, 1], alpha=0.25, s=18, color="gray", label="page embeddings")
        for i, (label, color) in enumerate(zip(snap_keys, colors)):
            ax.scatter(*snap_2d[i], s=160, color=color, zorder=5, label=f"centroid @ {label}")
            ax.annotate(str(label), snap_2d[i], textcoords="offset points", xytext=(6, 4), fontsize=9)

        ax.set_title("Topic Centroid Drift (PCA projection)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()
        print(f"  Saved PCA plot: {output}")

    except ImportError as e:
        print(f"  (Skipping PCA plot, missing library: {e})")


def _make_labelled_dataset(n=500, seed=42):
    relevant_templates = [
        "This paper proposes a {method} for {task} achieving state-of-the-art results on multiple benchmarks.",
        "We introduce a novel {method} architecture that significantly improves {task} performance.",
        "Empirical evaluation of {method} on {task} shows consistent gains over prior work.",
        "A comprehensive survey of {method} approaches for {task} in deep learning.",
        "Large-scale pre-training with {method} enables strong generalization on {task}.",
    ]
    irrelevant_templates = [
        "Best deals on {item}, up to {pct}% off at our online store this weekend.",
        "Local weather forecast: {weather} expected through the week with mild temperatures.",
        "The {team} won the championship game in a dramatic {pct}-point overtime victory.",
        "New restaurant opens downtown featuring {cuisine} fusion cuisine and craft cocktails.",
        "City council approves new zoning regulations for {district} residential development.",
    ]
    methods = ["transformer", "diffusion model", "GNN", "contrastive learning", "RL policy"]
    tasks = ["image classification", "NLP", "graph learning", "question answering", "generation"]
    items = ["sunglasses", "laptops", "sneakers", "furniture", "electronics"]
    weathers = ["sunny skies", "rain", "clouds", "fog", "thunderstorms"]
    teams = ["home team", "visiting side", "underdog"]
    cuisines = ["Mediterranean", "Asian", "Latin American", "French"]
    districts = ["downtown", "suburban", "riverside", "hillside"]

    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n):
        if i % 2 == 0:
            tmpl = rng.choice(relevant_templates)
            text = tmpl.format(method=rng.choice(methods), task=rng.choice(tasks)) + f" [sample {i}]"
            labels.append(1)
        else:
            tmpl = rng.choice(irrelevant_templates)
            text = tmpl.format(
                item=rng.choice(items), pct=rng.randint(20, 70),
                weather=rng.choice(weathers), team=rng.choice(teams),
                cuisine=rng.choice(cuisines), district=rng.choice(districts),
            ) + f" [sample {i}]"
            labels.append(0)
        texts.append(text)
    return texts, labels


def walkthrough_classifier(save_plot=True, output="classifier_eval.png"):
    _header("WALKTHROUGH 3: Relevance classifier training and evaluation")

    print("  Generating 500-page synthetic labeled dataset (50% relevant)...")
    texts, labels = _make_labelled_dataset(500)

    clf = RelevanceClassifier()
    print("  Training logistic regression on page embeddings (80/20 split)...\n")
    metrics = clf.fit(texts, labels, test_size=0.2, verbose=True)

    print("\n  Hold-out metrics:")
    for k, v in metrics.items():
        print(f"    {k:<12}: {v:.4f}")

    if not save_plot:
        return

    try:
        import matplotlib.pyplot as plt

        keys = ["precision", "recall", "f1", "accuracy", "roc_auc"]
        values = [metrics[k] for k in keys]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(keys, values, color="steelblue", alpha=0.85, edgecolor="white")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Relevance Classifier - Hold-out Evaluation Metrics")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()
        print(f"\n  Saved classifier evaluation chart: {output}")

    except ImportError as e:
        print(f"  (Skipping plot, missing library: {e})")


def walkthrough_frontier(prioritizer):
    _header("WALKTHROUGH 4: Semantic frontier integration (mock)")

    centroid = prioritizer.init_centroid(TOPIC_SEEDS)
    extractor = URLContextExtractor()
    frontier = MockFrontier(mode="priority")

    candidates = [
        ("Survey of deep learning methods 2023", "Recent advances in deep learning across vision, NLP, and speech. We survey 500+ papers.", "https://arxiv.org/abs/cs/deep-learning-survey-2023"),
        ("Click here", "See our full catalog of products and services. Great deals available now.", "https://shop.example.com/products/catalog"),
        ("Attention is all you need", "We propose a new sequence model architecture based entirely on attention mechanisms.", "https://papers.arxiv.org/abs/transformer-attention"),
        ("Weekend electronics sale", "Up to 70% off on laptops, tablets, and home appliances. Limited time only.", "https://deals.store.com/sale/electronics/weekend"),
        ("Neural architecture search tutorial", "Automated machine learning for discovering optimal neural network architectures efficiently.", "https://automl.org/nas/tutorial/introduction"),
        ("Reinforcement learning from human feedback", "RLHF enables alignment of large language models with human preferences and values.", "https://openai.com/research/rlhf-language-models"),
        ("Local restaurant reviews", "Find the best restaurants near you. Read reviews, see menus, and make reservations.", "https://yelp.com/search/restaurants/downtown"),
        ("Diffusion models beat GANs on image synthesis", "We show that denoising diffusion probabilistic models outperform GANs on FID score.", "https://arxiv.org/abs/diffusion-beats-gans-2021"),
    ]

    print("  Scoring and enqueuing candidate URLs:\n")
    print(f"  {'Score':>7}  URL")
    print(f"  {'-'*7}  {'-'*55}")
    for anchor, surrounding, url in candidates:
        ctx = extractor.extract(anchor, surrounding, url)
        score = prioritizer.score(ctx, centroid)
        frontier.push(url, priority=score, metadata={"anchor": anchor, "score": score})
        print(f"  {score:7.4f}  {url}")

    print(f"\n  Pop order (highest relevance score first):\n")
    rank = 1
    while not frontier.is_empty():
        url, priority, meta = frontier.pop()
        print(f"  #{rank:>2}  [{priority:.4f}]  {meta['anchor']!r}")
        rank += 1
    print()


def _header(title):
    print(f"\n{'=' * 62}")
    print(title)
    print("=" * 62)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walkthrough", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    save_plot = not args.no_plot

    print("Loading SemanticPrioritizer (all-MiniLM-L6-v2)...")
    prioritizer = SemanticPrioritizer()

    run_all = args.walkthrough is None

    if run_all or args.walkthrough == 1:
        walkthrough_score(prioritizer)
    if run_all or args.walkthrough == 2:
        walkthrough_centroid_drift(prioritizer, save_plot=save_plot)
    if run_all or args.walkthrough == 3:
        walkthrough_classifier(save_plot=save_plot)
    if run_all or args.walkthrough == 4:
        walkthrough_frontier(prioritizer)

    print("Done.")


if __name__ == "__main__":
    main()
