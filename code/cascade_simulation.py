#!/usr/bin/env python3
"""Cls-gated cascade simulation (CPU; device costs plugged in later).

Gate = benchmark yolo26n-cls 224 (per-frame P(rip) dumps in models/classification/.../predictions).
Detector = a student (per-frame COCO dumps). For gate thresholds swept on VAL (the detector's conf
is the val-frozen one from the eval TSV): frames with P(rip) < t get NO detections (gated out),
others keep the detector's boxes. Reports on TEST: F2@50 / P / R (benchmark matcher, bfi GT),
fraction of frames on which the detector ran, and the compute ratio
(gate_cost + frac_run * det_cost) / det_cost with GFLOPs as the stand-in cost (replaced by measured
mJ/frame on the device). Also a k-of-n temporal confirmation variant on the gate (frame order =
native index within a video). Rows -> results/cascade_sim.tsv. No CLI args. Env: conda yoloV26."""
import csv
import importlib.util
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
CLS_DIR = os.path.join(DATASET_ROOT, "models", "classification", "best_models", "predictions")
CLS_STEM = "3_cls_yolo26_nano_224_best_e50_j2856380"           # yolo26n-cls 224, canonical
DET_STEMS = {"dense_yolo26n_s1": ("/mnt/linux/icra_edge/predictions/students", "t1_nano_k025_t024_s1_n_s1_best_e2"),
             "canonical_yolo26n_s1": ("/mnt/linux/icra_edge/predictions/students", "canon_bfi_n_s1_best_e18")}
TASK = "bbox_from_instance"
GATE_THRESHOLDS = [round(x, 2) for x in np.arange(0.0, 0.96, 0.05)]
KOFN = [(1, 1), (2, 3), (3, 5)]         # k-of-n confirmation on the gate (n consecutive sampled frames)
GATE_GFLOPS, DET_GFLOPS = 0.6, 5.8      # yolo26n-cls 224 (approx.) / yolo26n det 640 - placeholders until measured
OUT_TSV = os.path.join(HERE, "results", "cascade_sim.tsv")
VIDEO_RE = re.compile(r"(RipBench-(?:NR-)?\d+)_(\d+)\.jpg$")


def load_ed():
    spec = importlib.util.spec_from_file_location("ed_cs", os.path.join(REPO_ROOT, "eval", "evaluate_detection.py"))
    ed = importlib.util.module_from_spec(spec)
    sys.modules["ed_cs"] = ed
    spec.loader.exec_module(ed)
    return ed


def det_conf(stem):
    for r in csv.DictReader(open(os.path.join(HERE, "results", "student_eval_bfi.tsv")), delimiter="\t"):
        if r["model"] == stem and r["split"] == "test":
            return float(r["selected_conf"])
    raise KeyError(stem)


def gate_mask(gt, cls_probs, thr, k, n):
    """{image_id: gate_open} with k-of-n confirmation along each video's frame order."""
    by_video = {}
    for im in gt.dataset["images"]:
        v, idx = VIDEO_RE.search(im["file_name"]).groups()
        by_video.setdefault(v, []).append((int(idx), im["id"], cls_probs[os.path.basename(im["file_name"])]))
    open_ = {}
    for v, frames in by_video.items():
        frames.sort()
        flags = [p >= thr for _, _, p in frames]
        for i, (_, iid, _) in enumerate(frames):
            window = flags[max(0, i - n + 1):i + 1]
            open_[iid] = sum(window) >= min(k, len(window)) if k > 1 else flags[i]
    return open_


def counts(ed, gt, preds, conf, gate_open):
    kept = [p for p in preds if p["score"] >= conf and gate_open.get(p["image_id"], True)]
    pred_coco = gt.loadRes(kept) if kept else None
    cache = ed.build_cache(gt, pred_coco, "bbox")
    m = ed.metrics_at(cache, conf, 0.50)
    return m


def main():
    ed = load_ed()
    gt_v = ed.COCO(os.path.join(DATASET_ROOT, "labels", TASK, "coco", "val.json"))
    gt_t = ed.COCO(os.path.join(DATASET_ROOT, "labels", TASK, "coco", "test.json"))
    cls_v = json.load(open(os.path.join(CLS_DIR, f"{CLS_STEM}_preds_val.json")))
    cls_t = json.load(open(os.path.join(CLS_DIR, f"{CLS_STEM}_preds_test.json")))
    write_header = not os.path.isfile(OUT_TSV)
    out = open(OUT_TSV, "a")
    if write_header:
        out.write("timestamp\tdetector\tgate\tk\tn\tgate_thr\tselected_on\tsplit\tF2_50\tP50\tR50\tfrac_detector_run\tcompute_ratio\tF2_always_on\n")
    for name, (pdir, stem) in DET_STEMS.items():
        conf = det_conf(stem)
        pv = json.load(open(os.path.join(pdir, f"{stem}_preds_val.json")))
        pt = json.load(open(os.path.join(pdir, f"{stem}_preds_test.json")))
        base_t = counts(ed, gt_t, pt, conf, {})
        for k, n in KOFN:
            # select the gate threshold on VAL: highest threshold whose val F2 stays within 0.005 of always-on
            base_v = counts(ed, gt_v, pv, conf, {})["F2"]
            best_thr = 0.0
            for thr in GATE_THRESHOLDS:
                f2v = counts(ed, gt_v, pv, conf, gate_mask(gt_v, cls_v, thr, k, n))["F2"]
                if f2v >= base_v - 0.005:
                    best_thr = thr
            for thr in sorted(set([best_thr] + [0.1, 0.3, 0.5])):
                go = gate_mask(gt_t, cls_t, thr, k, n)
                frac = float(np.mean(list(go.values())))
                m = counts(ed, gt_t, pt, conf, go)
                ratio = (GATE_GFLOPS + frac * DET_GFLOPS) / DET_GFLOPS
                out.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{name}\t{CLS_STEM}\t{k}\t{n}\t{thr:.2f}\t{'val' if thr == best_thr else 'fixed'}\ttest\t"
                          f"{m['F2']:.4f}\t{m['P']:.4f}\t{m['R']:.4f}\t{frac:.3f}\t{ratio:.3f}\t{base_t['F2']:.4f}\n")
                print(f"{name} gate {k}-of-{n} thr {thr:.2f}{' (val)' if thr == best_thr else ''}: F2 {m['F2']:.3f} "
                      f"(always-on {base_t['F2']:.3f}), detector runs on {frac:.1%} of frames, compute x{ratio:.2f}")
    out.close()
    print(f"-> {OUT_TSV}")


if __name__ == "__main__":
    main()
