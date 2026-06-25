import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent
RUNS = ROOT / "results" / "runs"
FIGURES = ROOT / "results" / "figures"

ORDER = ["drop_both_cams", "drop_wrist", "drop_exterior", "rand_proprio", "zero_proprio",
         "lang_empty", "lang_swap", "lang_gibberish"]
LABELS = {
    "drop_both_cams": "drop both cams", "drop_wrist": "drop wrist", "drop_exterior": "drop exterior",
    "rand_proprio": "random proprio", "zero_proprio": "zero proprio", "lang_empty": "remove instruction",
    "lang_swap": "swap instruction", "lang_gibberish": "gibberish instruction",
}


def load(*names):
    effects = {c: [] for c in ORDER}
    floors = []
    for name in names:
        data = np.load(RUNS / f"{name}.npz", allow_pickle=True)
        effect, floor = data["effects"].item(), data["floors"]
        floors.append(float(floor.mean()))
        for c in ORDER:
            effects[c].append(float(np.mean(effect[c] / floor)))
    return {c: float(np.mean(effects[c])) for c in ORDER}


def grouped_bars(ax, series, colors):
    x = np.arange(len(ORDER))
    width = 0.8 / len(series)
    for i, (label, values) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        rects = ax.bar(x + offset, [max(values[c], 0.02) for c in ORDER], width, label=label, color=colors[i])
        for rect, c in zip(rects, ORDER):
            value = values[c]
            ax.text(rect.get_x() + rect.get_width() / 2, max(value, 0.02) * 1.06,
                    "0" if value < 0.01 else f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, ls="--", color="gray")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in ORDER], rotation=20, ha="right")


def comparison():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    grouped_bars(
        ax,
        [("pi05_libero (sim)", load("libero_ep0", "libero_ep6")),
         ("pi05_droid (real)", load("droid_ep0", "droid_ep1"))],
        ["#1f77b4", "#ff7f0e"],
    )
    ax.set_ylabel("action change (relative to the model's own sampling noise, log)")
    ax.set_title("Which inputs drive the action? pi05_libero vs pi05_droid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "libero_vs_droid.png", dpi=130)


def cross_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    grouped_bars(
        axes[0],
        [("in-distribution (sim)", load("libero_ep0", "libero_ep6")),
         ("DROID frames (real, OOD)", load("cross_libero_on_droid"))],
        ["#1f77b4", "#d62728"],
    )
    grouped_bars(
        axes[1],
        [("in-distribution (real)", load("droid_ep0", "droid_ep1")),
         ("LIBERO frames (sim, OOD)", load("cross_droid_on_libero"))],
        ["#ff7f0e", "#d62728"],
    )
    axes[0].set_title("pi05_libero")
    axes[1].set_title("pi05_droid")
    axes[0].set_ylabel("action change (relative to sampling noise, log)")
    for ax in axes:
        ax.legend()
    fig.suptitle("Cross-distribution: each policy keeps its fingerprint on the other domain's inputs")
    fig.tight_layout()
    fig.savefig(FIGURES / "cross_distribution.png", dpi=130)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    comparison()
    cross_distribution()
    print(f"wrote figures to {FIGURES}")
