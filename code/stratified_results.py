#!/usr/bin/env python3
"""Per-stratum test results (ICRA): where does dense supervision help?

Joins per-video TP/FP/FN (test, bbox_from_instance GT, conf frozen on val exactly as the
benchmark evaluator does; matcher imported from eval/evaluate_detection.py through
bootstrap_cis.py) with the per-video metadata sheet metadata/RipBench_video_annotation_data.csv.
For every stratum of every STRATA column: n videos, pooled F2 per VARIANT (seeds micro-pooled,
as in RESULTS.md's seed-pooled rows) and the paired dense - canonical delta with a video-level
bootstrap CI inside the stratum (B draws, seed 0). Rows -> results/stratified_test.tsv.
Constants below, no CLI args. Env: conda yoloV26 (CPU only).
"""
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bootstrap_cis as bc  # noqa: E402  (loaders + per-video counts; PAIRS there are not used)

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
META_CSV = os.path.join(REPO_ROOT, "metadata", "RipBench_video_annotation_data.csv")
TASK = "bbox_from_instance"
VARIANTS = {  # name -> regex on the dump stem (all seeds pooled); order = table order
    "yolo26n canonical": r"^canon_bfi_n_s\d+_best_e\d+$",
    "yolo26n interp": r"^interp_gt_s1_n_s\d+_best_e\d+$",
    "yolo26n dense": r"^t1_nano_k025_t024_s1_n_s\d+_best_e\d+$",
    "yolo26s canonical": r"^canon_bfi_s_s\d+_best_e\d+$",
    "yolo26s dense": r"^t1_nano_k025_t024_s1_s_s\d+_best_e\d+$",
}
DELTA_PAIRS = [("yolo26n canonical", "yolo26n dense"), ("yolo26n interp", "yolo26n dense"),
               ("yolo26s canonical", "yolo26s dense")]
STRATA = ["Viewpoint", "Camera motion type", "Visual signature (primary)", "Shoreline visibility", "Rip present"]
B, SEED = 1000, 0
OUT_TSV = os.path.join(HERE, "results", "stratified_test.tsv")


def find_stems():
    stems = {}
    for name, rx in VARIANTS.items():
        found = []
        for d in bc.PRED_DIRS:
            for p in glob.glob(os.path.join(d, "*_preds_test.json")):
                stem = os.path.basename(p)[:-len("_preds_test.json")]
                if re.match(rx, stem) and os.path.isfile(os.path.join(d, f"{stem}_preds_val.json")):
                    found.append((stem, d))
        stems[name] = sorted(set(found))
    return stems


def pooled_counts(ed, gt_v, gt_t, runs):
    """Per-video counts summed over seeds (each seed at its own val-frozen conf)."""
    tot = {}
    for stem, d in runs:
        pv = json.load(open(os.path.join(d, f"{stem}_preds_val.json")))
        pt = json.load(open(os.path.join(d, f"{stem}_preds_test.json")))
        conf = bc.select_conf(ed, gt_v, pv)
        c = bc.per_video_counts(ed, gt_t, pt, conf)
        for v, x in c.items():
            t = tot.setdefault(v, [0, 0, 0])
            for i in range(3):
                t[i] += x[i]
        print(f"  {stem}: conf {conf:.2f}", flush=True)
    return tot


def f2(counts, videos):
    return bc.f2_from(counts, range(len(videos)), videos)


def main():
    ed = bc.load_ed()
    base = os.path.join(bc.DATASET_ROOT, "labels", TASK, "coco")
    gt_v, gt_t = ed.COCO(os.path.join(base, "val.json")), ed.COCO(os.path.join(base, "test.json"))
    meta = {r["Rip name"]: r for r in csv.DictReader(open(META_CSV))}
    stems = find_stems()
    for k, v in stems.items():
        print(f"{k}: {len(v)} runs", flush=True)
    counts = {name: pooled_counts(ed, gt_v, gt_t, runs) for name, runs in stems.items() if runs}
    videos_all = sorted(next(iter(counts.values())))
    rng = np.random.default_rng(SEED)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for col in STRATA:
        values = sorted({(meta[v][col] or "n/a") for v in videos_all})
        for val in values + ["ALL"]:
            vids = [v for v in videos_all if val == "ALL" or (meta[v][col] or "n/a") == val]
            n_rip = sum(1 for v in vids if "NR-" not in v)
            row = {"timestamp": ts, "task": TASK, "stratum_col": col, "stratum": val,
                   "n_videos": len(vids), "n_rip_videos": n_rip,
                   "n_gt_boxes": sum(counts[next(iter(counts))][v][0] + counts[next(iter(counts))][v][2] for v in vids)}
            for name, c in counts.items():
                row[f"F2_{name}"] = f"{f2(c, vids):.4f}"
            for a, b_ in DELTA_PAIRS:
                if a not in counts or b_ not in counts:
                    continue
                d = f2(counts[b_], vids) - f2(counts[a], vids)
                idx = rng.integers(0, len(vids), size=(B, len(vids)))
                boots = np.array([bc.f2_from(counts[b_], i, vids) - bc.f2_from(counts[a], i, vids) for i in idx])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                row[f"dF2 {b_} - {a}"] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            rows.append(row)
            print(f"{col} = {val}: n={len(vids)} " + " ".join(f"{k.split('_',1)[1]}={v}" for k, v in row.items() if k.startswith("F2_")), flush=True)
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    with open(OUT_TSV, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    print("wrote", OUT_TSV)


if __name__ == "__main__":
    main()
