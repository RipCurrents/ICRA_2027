#!/usr/bin/env python3
"""Label-spacing sweep figure: test F2 vs expert-annotation spacing (SDF interpolation between anchors).

Points: interp_gt_a{2,4,8,16} (1 seed each) from results/student_eval_bfi.tsv; references: full interpolation (interp_gt_s1,
3 seeds: mean + min-max band) and sparse-only (canon_bfi_n, 3 seeds: mean + band). x = seconds between expert anchors at
30 fps (0.2·k). Palette = dataviz reference palette. -> ICRA_2027/paper/figures/spacing_sweep.{pdf,png}. No CLI args.
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TSV = os.path.join(HERE, "results", "student_eval_bfi.tsv")
FIG_DIR = os.path.join(REPO_ROOT, "ICRA_2027", "paper", "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})


def main():
    rows = [r for r in csv.DictReader(open(TSV), delimiter="\t") if r["split"] == "test"]
    f2 = lambda rx: [float(r["F2_50"]) for r in rows if re.match(rx, r["model"])]
    sweep = {}
    for r in rows:
        m = re.match(r"^interp_gt_a(\d+)_n_s1_best_e\d+$", r["model"])
        if m:
            sweep[int(m.group(1))] = float(r["F2_50"])
    full = f2(r"^interp_gt_s1_n_s\d+_best_e\d+$")
    sparse = f2(r"^canon_bfi_n_s\d+_best_e\d+$")
    ks = sorted(sweep)
    xs = [0.2 * k for k in ks]
    fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=200)
    ax.axhspan(min(full), max(full), color=BLUE, alpha=0.12, lw=0, zorder=0)
    ax.axhline(np.mean(full), color=BLUE, lw=1.2, ls="--", zorder=1)
    ax.axhspan(min(sparse), max(sparse), color=ORANGE, alpha=0.12, lw=0, zorder=0)
    ax.axhline(np.mean(sparse), color=ORANGE, lw=1.2, ls=":", zorder=1)
    ax.plot(xs, [sweep[k] for k in ks], color=AQUA, lw=1.5, zorder=2)
    ax.scatter(xs, [sweep[k] for k in ks], s=28, color=AQUA, edgecolor="white", linewidth=0.6, zorder=3)
    for k, x in zip(ks, xs):
        ax.annotate(f"1/{k}", (x, sweep[k]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6, color=MUTED)
    ax.set_xscale("log"); ax.set_xticks([0.2, 0.4, 0.8, 1.6, 3.2]); ax.set_xticklabels(["0.2", "0.4", "0.8", "1.6", "3.2"])
    ax.set_xlabel("seconds between expert-annotated frames")
    ax.set_ylabel("test F2@0.5 (yolo26n)")
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend([Line2D([], [], color=BLUE, ls="--"), Patch(color=BLUE, alpha=0.2), Line2D([], [], color=ORANGE, ls=":"),
               Line2D([], [], color=AQUA, marker="o", markersize=4)],
              ["full interpolation (3-seed mean)", "3-seed range", "expert only (3 seeds)", "anchor sweep (1 seed)"],
              fontsize=5.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"spacing_sweep.{ext}"), bbox_inches="tight", pad_inches=0.02)
    print("sweep:", {0.2 * k: sweep[k] for k in ks}, "full", np.round(full, 3), "sparse", np.round(sparse, 3))


if __name__ == "__main__":
    main()
