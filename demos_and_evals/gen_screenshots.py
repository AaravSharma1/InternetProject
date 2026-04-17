import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG     = "#1E1E2E"
GREEN  = "#A6E3A1"
YELLOW = "#F9E2AF"
RED    = "#F38BA8"
BLUE   = "#89B4FA"
GRAY   = "#6C7086"
WHITE  = "#CDD6F4"
CYAN   = "#89DCEB"

def make_slide1(output_path):
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    lines = [
        (0.03, 0.93, "$ python3 demo.py --walkthrough 1", CYAN,  11, "monospace"),
        (0.03, 0.85, "Topic centroid initialized from seed descriptions:", WHITE, 10, "monospace"),
        (0.05, 0.79, '"machine learning research papers"', YELLOW, 10, "monospace"),
        (0.05, 0.73, '"deep learning neural networks survey"', YELLOW, 10, "monospace"),
        (0.05, 0.67, '"artificial intelligence academic publications"', YELLOW, 10, "monospace"),

        (0.03, 0.58, "-- BEST CASE ---------------------------------------------------", GREEN, 10, "monospace"),
        (0.03, 0.52, "  anchor  : 'deep learning survey paper NeurIPS 2023'", WHITE, 10, "monospace"),
        (0.03, 0.46, "  url     : arxiv.org/abs/cs/deep-learning-survey-neurips2023", WHITE, 10, "monospace"),
        (0.03, 0.40, "  score   : 0.5664  ->  top of frontier, fetched immediately", GREEN, 10.5, "monospace"),

        (0.03, 0.31, "-- AVERAGE CASE ------------------------------------------------", YELLOW, 10, "monospace"),
        (0.03, 0.25, "  anchor  : 'click here'", WHITE, 10, "monospace"),
        (0.03, 0.19, "  url     : cs.university.edu/faculty/profile/homepage", WHITE, 10, "monospace"),
        (0.03, 0.13, "  score   : 0.2393  ->  mid-queue, fetched if budget allows", YELLOW, 10.5, "monospace"),

        (0.50, 0.58, "-- WORST CASE --------------------------------------------------", RED, 10, "monospace"),
        (0.50, 0.52, "  anchor  : 'Buy now, 50% off!'", WHITE, 10, "monospace"),
        (0.50, 0.46, "  url     : shop.deals.com/sale/accessories/sunglasses-2024", WHITE, 10, "monospace"),
        (0.50, 0.40, "  score   : -0.0075  ->  bottom of frontier, never fetched", RED, 10.5, "monospace"),

        (0.50, 0.28, "score() pipeline:", GRAY, 10, "monospace"),
        (0.50, 0.22, "  anchor + surrounding text + URL tokens", WHITE, 10, "monospace"),
        (0.50, 0.16, "       |  all-MiniLM-L6-v2  (384-dim embedding)", WHITE, 10, "monospace"),
        (0.50, 0.10, "       |  cosine_similarity(embedding, topic_centroid)", WHITE, 10, "monospace"),
        (0.50, 0.04, "       ->  score in [-1, 1]   (higher = more relevant)", GREEN, 10, "monospace"),
    ]

    for x, y, text, color, size, family in lines:
        ax.text(x, y, text, color=color, fontsize=size,
                fontfamily=family, transform=ax.transAxes,
                verticalalignment="center")

    bar = mpatches.FancyBboxPatch((0, 0.97), 1, 0.03,
        boxstyle="square,pad=0", facecolor="#313244", transform=ax.transAxes, clip_on=False)
    ax.add_patch(bar)
    ax.text(0.5, 0.985, "demo.py  --  walkthrough 1: score() end-to-end",
            color=GRAY, fontsize=9, fontfamily="monospace",
            transform=ax.transAxes, ha="center", va="center")

    ax.plot([0.485, 0.485], [0.0, 0.68], color=GRAY, linewidth=0.6, alpha=0.4,
            transform=ax.transAxes)

    plt.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {output_path}")

def make_slide2(output_path):
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # Title bar
    bar = mpatches.FancyBboxPatch((0, 0.97), 1, 0.03,
        boxstyle="square,pad=0", facecolor="#313244", transform=ax.transAxes, clip_on=False)
    ax.add_patch(bar)
    ax.text(0.5, 0.985, "demo.py  --  walkthrough 4: priority frontier ordering",
            color=GRAY, fontsize=9, fontfamily="monospace",
            transform=ax.transAxes, ha="center", va="center")

    # Command line
    ax.text(0.03, 0.93, "$ python3 demo.py --walkthrough 4", color=CYAN, fontsize=11,
            fontfamily="monospace", transform=ax.transAxes, va="center")
    ax.text(0.03, 0.86, "8 URLs discovered during crawl. Semantic scores assigned:", color=WHITE,
            fontsize=10, fontfamily="monospace", transform=ax.transAxes, va="center")

    # URL score table
    entries = [
        ("0.5158", "arxiv.org",        "'Survey of deep learning methods 2023'",         GREEN),
        ("0.4512", "automl.org",       "'Neural architecture search tutorial'",           GREEN),
        ("0.2757", "openai.com",       "'Reinforcement learning from human feedback'",    YELLOW),
        ("0.2622", "papers.arxiv.org", "'Attention is all you need'",                     YELLOW),
        ("0.2270", "arxiv.org",        "'Diffusion models beat GANs'",                    YELLOW),
        ("0.1174", "shop.example.com", "'Click here'",                                    RED),
        ("0.0707", "deals.store.com",  "'Weekend electronics sale'",                      RED),
        ("0.0246", "yelp.com",         "'Local restaurant reviews'",                      RED),
    ]

    y_start = 0.78
    row_h   = 0.076
    for i, (score, domain, anchor, color) in enumerate(entries):
        y = y_start - i * row_h
        rank = f"#{i+1}"
        ax.text(0.03,  y, rank,   color=GRAY,  fontsize=10, fontfamily="monospace", transform=ax.transAxes, va="center")
        ax.text(0.075, y, score,  color=color, fontsize=10, fontfamily="monospace", transform=ax.transAxes, va="center")
        ax.text(0.155, y, f"{domain:<22}", color=WHITE, fontsize=10, fontfamily="monospace", transform=ax.transAxes, va="center")
        ax.text(0.42,  y, anchor, color=GRAY,  fontsize=9.5, fontfamily="monospace", transform=ax.transAxes, va="center")

    # Column headers
    ax.text(0.075, 0.855, "score ", color=GRAY, fontsize=9, fontfamily="monospace", transform=ax.transAxes, va="center")
    ax.text(0.155, 0.855, "domain                ", color=GRAY, fontsize=9, fontfamily="monospace", transform=ax.transAxes, va="center")
    ax.text(0.42,  0.855, "anchor text", color=GRAY, fontsize=9, fontfamily="monospace", transform=ax.transAxes, va="center")
    ax.plot([0.03, 0.97], [0.842, 0.842], color=GRAY, linewidth=0.6, alpha=0.5, transform=ax.transAxes)

    # Pop order summary
    ax.plot([0.03, 0.97], [0.18, 0.18], color=GRAY, linewidth=0.6, alpha=0.4, transform=ax.transAxes)
    ax.text(0.03, 0.14, "Frontier pop order (min-heap, lower value = higher priority):", color=WHITE,
            fontsize=10, fontfamily="monospace", transform=ax.transAxes, va="center")
    ax.text(0.03, 0.07, "  #1 Survey of deep learning...   #2 Neural architecture search...   ...   #8 Local restaurant reviews",
            color=GREEN, fontsize=9.5, fontfamily="monospace", transform=ax.transAxes, va="center")
    ax.text(0.03, 0.01, "  BFS would have popped these in arbitrary insertion order — semantic mode fetches relevant pages first",
            color=GRAY, fontsize=9, fontfamily="monospace", transform=ax.transAxes, va="center")

    plt.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {output_path}")


make_slide1("/Users/aaravsharma1/InternetProject/demos_and_evals/walkthrough1_screenshot.png")
make_slide2("/Users/aaravsharma1/InternetProject/demos_and_evals/walkthrough2_screenshot.png")
