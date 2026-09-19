#!/usr/bin/env python3
"""Accuracy-vs-latency (and -energy) Pareto figures for the ICRA paper, regenerated from the result TSVs.

Points = every (student checkpoint, input size, precision, engine) that has BOTH a benchmark-evaluator accuracy row
(test F2@50 on the manual TIGHT box view, so the RF-DETR-n heavy reference is comparable) and a latency row.
Latency source: results/device_latency.tsv (Jetson, TensorRT) when present, else the RTX-4090 onnxruntime stand-in
results/latency_4090.tsv (headerless; columns below) - the figure says which. Energy figure only from
results/device_energy.tsv (mJ per frame). Seeds: the first seed of each variant is plotted (latency does not depend
on the seed); accuracy = that checkpoint's own row. Heavy reference: RF-DETR-n (results/reference_eval_tight.tsv +
eval/results/fps_operating.tsv read-only for the 4090 stand-in). Palette = the dataviz reference palette (validated).
Constants below, no CLI args. Env: conda yoloV26 (matplotlib). Output: ICRA_2027/paper/figures/pareto_*.{pdf,png}.
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(HERE, "results")
FIG_DIR = os.path.join(REPO_ROOT, "ICRA_2027", "paper", "figures")
ACC_TSV = os.path.join(RES, "student_eval_tight.tsv")          # tight view = same GT as the heavy reference
REF_TSV = os.path.join(RES, "reference_eval_tight.tsv")
DEVICE_LAT = os.path.join(RES, "device_latency.tsv")
DEVICE_EN = os.path.join(RES, "device_energy.tsv")
STANDIN_LAT = os.path.join(RES, "latency_4090.tsv")
STANDIN_COLS = ["timestamp", "model", "backend", "precision", "imgsz", "n_timed", "repeats", "ms_pre", "ms_infer", "ms_post", "fps", "device"]
FPS_OPERATING = os.path.join(REPO_ROOT, "eval", "results", "fps_operating.tsv")   # read-only benchmark table
VARIANTS = [  # (label, stem regex of the FIRST seed checkpoint, marker)
    ("yolo26n, expert labels", r"^canon_bfi_n_s1_best_e\d+$", "o"),
    ("yolo26n, dense", r"^t1_nano_k025_t024_s1_n_s1_best_e\d+$", "s"),
    ("yolo26n, dense + smoothing", r"^t1_nano_k025_t024_s1_n_s1_best_e\d+_sm2$", "P"),
    ("yolo26s, dense", r"^t1_nano_k025_t024_s1_s_s[13]_best_e\d+$", "D"),   # 4090 rows use seed 1, device rows seed 3
]
REFERENCE = ("RF-DETR-n (30 M)", r"^1_det_rfdetr_nano_best_best_ema_j3178618(_onnx)?$", "^")
PALETTE = {"fp32": "#2a78d6", "fp16": "#eb6834", "int8": "#1baf7a"}     # dataviz reference palette slots 1-3
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})


def read_tsv(path, cols=None):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        if cols:
            return [dict(zip(cols, l.rstrip("\n").split("\t"))) for l in f if l.strip()]
        return list(csv.DictReader(f, delimiter="\t"))


def accuracy_rows():
    acc = {}
    for r in read_tsv(ACC_TSV) + read_tsv(REF_TSV):
        if r["split"] == "test":
            acc[r["model"]] = float(r["F2_50"])
    return acc


def base_stem(model):
    """'<stem>_img512' -> ('<stem>', 512); plain -> (stem, 640); the _sm2 suffix stays part of the stem."""
    m = re.match(r"^(.*)_img(\d+)$", model)
    return (m.group(1), int(m.group(2))) if m else (model, 640)


def latency_points():
    """[(stem, imgsz, precision, engine, ms, source)]"""
    pts, src = [], None
    dev = read_tsv(DEVICE_LAT)
    if dev:
        src = f"{dev[0]['device']} ({dev[0]['engine']}, {dev[0]['power_mode']})"
        for r in dev:
            if r["precision"] == "int8" or "trtexec" in r["engine"]:   # one timing boundary per frontier: in-process GPU time only (INT8 = trtexec rows, kept in the table)
                continue
            pts.append((r["model"], int(r["imgsz"]), r["precision"], r["engine"], float(r["infer_ms"]), float(r["F2_device"]) if r.get("F2_device") else None))
    else:
        src = "RTX 4090 stand-in (onnxruntime CUDA, batch 1)"
        for r in read_tsv(STANDIN_LAT, STANDIN_COLS):
            pts.append((r["model"], int(r["imgsz"]), r["precision"], r["backend"], float(r["ms_infer"])))
        for r in read_tsv(FPS_OPERATING):
            if r["model"].startswith("1_det_rfdetr_nano"):
                pts.append((r["model"] + "_onnx", int(r["imgsz"]), "fp32", "pytorch", float(r["ms_inference"])))
    return pts, src


def pareto(points):
    """points: [(x, y)] -> indices on the upper-left front (min x, max y)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best = [], -1
    for i in order:
        if points[i][1] > best:
            front.append(i); best = points[i][1]
    return front


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    acc = accuracy_rows()
    lat, src = latency_points()
    fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=200)
    xy, seen = [], set()
    for label, rx, marker in VARIANTS + [REFERENCE]:
        for row in lat:
            stem, imgsz, prec, engine, ms = row[:5]
            f2_dev = row[5] if len(row) > 5 else None
            cands = [stem] + ([stem + "_sm2"] if label.endswith("smoothing") else [])
            hit = [c for c in cands if re.match(rx, c)]
            if not hit:
                continue
            c = hit[0]
            if f2_dev is not None:                     # device rows carry their own dump accuracy (tight GT)
                if label.endswith("smoothing"):        # device dumps are unsmoothed - never plot them under that label
                    continue
                f2 = f2_dev
            else:
                key = c if (imgsz == 640 or label == REFERENCE[0]) else f"{c}_img{imgsz}"
                if key not in acc:
                    continue
                f2 = acc[key]
            color = PALETTE.get(prec, MUTED)
            ax.scatter(ms, f2, s=26, marker=marker, facecolor=color, edgecolor="white", linewidth=0.6, zorder=3,
                       label=label if label not in seen else None)
            seen.add(label)
            xy.append((ms, f2))
            if imgsz != 640 or label == REFERENCE[0]:
                ax.annotate(f"{imgsz}", (ms, f2), textcoords="offset points", xytext=(4, -2), fontsize=6, color=MUTED)
    if xy:
        fr = pareto(xy)
        ax.plot([xy[i][0] for i in fr], [xy[i][1] for i in fr], color=GRID, lw=1, zorder=2)
    ax.set_xscale("log")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.ticklabel_format(axis="x", style="plain")
    ax.set_xlabel("batch-1 inference latency [ms]")
    ax.set_title("Jetson Orin Nano Super (MAXN)", fontsize=6, color=MUTED, loc="left")
    ax.set_ylabel("test F2@0.5 (tight-box GT)")
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    prec_handles = [Line2D([], [], marker="o", ls="", color=c, markersize=4, label=p.upper()) for p, c in PALETTE.items() if any(pt[2] == p for pt in lat)]
    ax.legend(handles + prec_handles, labels + [h.get_label() for h in prec_handles], fontsize=5.7, frameon=False, loc="upper left", bbox_to_anchor=(0.00, 0.90), ncol=1, columnspacing=0.8, handletextpad=0.5, markerscale=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"pareto_latency.{ext}"))
    print(f"pareto_latency: {len(xy)} points, source = {src}")
    en = read_tsv(DEVICE_EN)
    if en:
        fig2, ax2 = plt.subplots(figsize=(3.4, 2.6), dpi=200)
        pts2 = []
        for label, rx, marker in VARIANTS + [REFERENCE]:
            for r in en:
                if re.match(rx, r["model"]):
                    key = r["model"] if int(r["imgsz"]) == 640 else f"{r['model']}_img{r['imgsz']}"
                    if key in acc:
                        ax2.scatter(float(r["mJ_per_frame"]), acc[key], s=26, marker=marker, facecolor=PALETTE.get(r["precision"], MUTED),
                                    edgecolor="white", linewidth=0.6, zorder=3, label=label)
                        pts2.append((float(r["mJ_per_frame"]), acc[key]))
        if pts2:
            fr = pareto(pts2)
            ax2.plot([pts2[i][0] for i in fr], [pts2[i][1] for i in fr], color=GRID, lw=1)
        ax2.set_xscale("log"); ax2.set_xlabel("energy per frame [mJ] (INA3221 rails)"); ax2.set_ylabel("test F2@0.5 (tight-box GT)")
        ax2.grid(True, color=GRID, lw=0.5, zorder=0)
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(os.path.join(FIG_DIR, f"pareto_energy.{ext}"))
        print(f"pareto_energy: {len(pts2)} points")
    else:
        print("no device_energy.tsv yet - energy figure skipped")


if __name__ == "__main__":
    main()
