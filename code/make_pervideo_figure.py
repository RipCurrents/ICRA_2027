#!/usr/bin/env python3
"""Per-video test F2 distribution next to the pooled F2 (critic: the pooled +0.05 is box-weighted).

Reads results/pervideo_f2_test.tsv (seeds micro-pooled) and draws, per variant, one dot per rip test video (31), the
per-video median (bar) and the pooled aggregate F2 (diamond) -> ICRA_2027/paper/figures/pervideo_strip.{pdf,png}.
Palette = dataviz reference palette. Constants below, no CLI args. Env: conda yoloV26 (matplotlib).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TSV = os.path.join(HERE, "results", "pervideo_f2_test.tsv")
FIG_DIR = os.path.join(REPO_ROOT, "ICRA_2027", "paper", "figures")
ORDER = ["yolo26n canonical", "yolo26n dense", "yolo26n dense + sm2", "yolo26s dense"]
LABELS = {"yolo26n canonical": "yolo26n\nexpert", "yolo26n dense": "yolo26n\ndense", "yolo26n dense + sm2": "yolo26n\ndense+sm", "yolo26s dense": "yolo26s\ndense"}
DOT, MEDIAN, POOLED = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})


def main():
    rows = list(csv.DictReader(open(TSV), delimiter="\t"))
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=200)
    rng = np.random.default_rng(0)
    for i, name in enumerate(ORDER):
        rip = [r for r in rows if r["variant"] == name and r["rip"] == "yes"]
        f2 = np.array([float(r["F2"]) for r in rip])
        tp = sum(int(r["TP"]) for r in rows if r["variant"] == name)
        fp = sum(int(r["FP"]) for r in rows if r["variant"] == name)
        fn = sum(int(r["FN"]) for r in rows if r["variant"] == name)
        p, rc = tp / (tp + fp), tp / (tp + fn)
        pooled = 5 * p * rc / (4 * p + rc)
        x = i + rng.uniform(-0.18, 0.18, size=len(f2))
        ax.scatter(x, f2, s=9, color=DOT, alpha=0.7, edgecolor="none", zorder=3)
        ax.hlines(np.median(f2), i - 0.3, i + 0.3, color=MEDIAN, lw=2, zorder=4)
        ax.scatter([i], [pooled], marker="D", s=28, color=POOLED, edgecolor="white", linewidth=0.6, zorder=5)
        ax.annotate(f"{np.median(f2):.2f}", (i + 0.32, np.median(f2)), fontsize=6, color=MEDIAN, va="center")
        ax.annotate(f"{pooled:.2f}", (i + 0.32, pooled), fontsize=6, color=POOLED, va="center")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LABELS[n] for n in ORDER], fontsize=7)
    ax.set_ylabel("test F2@0.5 per rip video")
    ax.set_ylim(-0.02, 1.02); ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker="o", ls="", color=DOT, markersize=3), Line2D([], [], color=MEDIAN, lw=2),
               Line2D([], [], marker="D", ls="", color=POOLED, markersize=4)],
              ["one test video (31)", "median per video", "pooled F2 (Table I)"], fontsize=6, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"pervideo_strip.{ext}"))
    print("wrote pervideo_strip.{pdf,png}")


if __name__ == "__main__":
    main()
