#!/usr/bin/env python3
"""Per-video test F2 table (provenance for the Limitations paragraph: per-video skew).

For every VARIANT (seeds micro-pooled per video, conf frozen on val per seed exactly as the evaluator does) writes one
row per test video: TP/FP/FN and F2 at IoU 0.5 (bfi GT), plus the video's metadata (viewpoint, camera motion, signature)
-> results/pervideo_f2_test.tsv, and prints the summary numbers quoted in the paper (median per-video F2 over the rip
videos, number of rip videos above 0.85, no-rip videos with FPs). Constants below, no CLI args. Env: conda yoloV26 (CPU).
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
import bootstrap_cis as bc  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
META_CSV = os.path.join(REPO_ROOT, "metadata", "RipBench_video_annotation_data.csv")
TASK = "bbox_from_instance"
VARIANTS = {"yolo26n canonical": r"^canon_bfi_n_s\d+_best_e\d+$",
            "yolo26n dense": r"^t1_nano_k025_t024_s1_n_s\d+_best_e\d+$",
            "yolo26n dense + sm2": r"^t1_nano_k025_t024_s1_n_s\d+_best_e\d+_sm2$",
            "yolo26s dense": r"^t1_nano_k025_t024_s1_s_s\d+_best_e\d+$"}
OUT_TSV = os.path.join(HERE, "results", "pervideo_f2_test.tsv")


def main():
    ed = bc.load_ed()
    base = os.path.join(bc.DATASET_ROOT, "labels", TASK, "coco")
    gt_v, gt_t = ed.COCO(os.path.join(base, "val.json")), ed.COCO(os.path.join(base, "test.json"))
    meta = {r["Rip name"]: r for r in csv.DictReader(open(META_CSV))}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for name, rx in VARIANTS.items():
        runs = sorted({(os.path.basename(p)[:-len("_preds_test.json")], d) for d in bc.PRED_DIRS
                       for p in glob.glob(os.path.join(d, "*_preds_test.json"))
                       if re.match(rx, os.path.basename(p)[:-len("_preds_test.json")])
                       and os.path.isfile(os.path.join(d, os.path.basename(p).replace("_test.json", "_val.json")))})
        tot = {}
        for stem, d in runs:
            conf = bc.select_conf(ed, gt_v, json.load(open(os.path.join(d, f"{stem}_preds_val.json"))))
            for v, c in bc.per_video_counts(ed, gt_t, json.load(open(os.path.join(d, f"{stem}_preds_test.json"))), conf).items():
                t = tot.setdefault(v, [0, 0, 0])
                for i in range(3):
                    t[i] += c[i]
        f2s = []
        for v in sorted(tot):
            tp, fp, fn = tot[v]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
            m = meta.get(v, {})
            rows.append(dict(timestamp=ts, variant=name, n_seeds=len(runs), video=v, rip="no" if "NR-" in v else "yes",
                             TP=tp, FP=fp, FN=fn, F2=f"{f2:.4f}", viewpoint=m.get("Viewpoint", ""),
                             camera_motion=m.get("Camera motion type", ""), signature=m.get("Visual signature (primary)", "")))
            if "NR-" not in v:
                f2s.append(f2)
        nr_fp = sum(1 for v in tot if "NR-" in v and tot[v][1] > 0)
        print(f"{name} ({len(runs)} seeds): rip videos median F2 {np.median(f2s):.3f}, mean {np.mean(f2s):.3f}, "
              f"n>0.85 = {sum(f > 0.85 for f in f2s)}, n<0.10 = {sum(f < 0.10 for f in f2s)} of {len(f2s)}; "
              f"no-rip videos with >=1 FP: {nr_fp}/{sum(1 for v in tot if 'NR-' in v)}", flush=True)
    with open(OUT_TSV, "w") as f:
        f.write("\t".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r.values()) + "\n")
    print("wrote", OUT_TSV, len(rows), "rows")


if __name__ == "__main__":
    main()
