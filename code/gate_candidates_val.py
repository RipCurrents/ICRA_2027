#!/usr/bin/env python3
"""VAL-ONLY gate comparison at the EVENT level (the author 2026-09-07: is a bigger / 640 px classifier a better gate?).
For each candidate classifier (benchmark val predictions, ultralytics = training preprocessing) and gate threshold,
replay the cascade on VAL with the dense yolo26n's val dump at conf 0.04, 1 Hz, k=n=1: rip-video recall vs no-rip
false-episode fraction vs detector duty. Nothing touches test. Constants at top, no CLI args. Env: yoloV26."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
GT_VAL = os.path.join(DATASET_ROOT, "labels", "bbox_from_instance", "coco", "val.json")
CLS_DIR = os.path.join(DATASET_ROOT, "models", "classification", "best_models", "predictions")
CANDIDATES = {"yolo26n-224 (current)": "3_cls_yolo26_nano_224_best_e50_j2856380",
              "yolo26s-224": "7_cls_yolo26_small_224_best_e35_j2856386",
              "yolo26m-224": "11_cls_yolo26_medium_224_best_e39_j2856390",
              "yolo26x-224": None,   # resolved by glob below
              "resnet34-640": "7_cls_resnet_resnet34_640_best_e19_j2857467",
              "yolo26n-640": None}
DET_VAL = "/mnt/linux/icra_edge/predictions/students/t1_nano_k025_t024_s1_n_s1_best_e2_preds_val.json"
DET_CONF = 0.04
THRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DUTY, K, N = 1.0, 1, 1
OUT_TSV = os.path.join(HERE, "results", "gate_candidates_val.tsv")


def main():
    import glob
    import event_metrics as em
    from pycocotools.coco import COCO
    for k, v in list(CANDIDATES.items()):
        if v is None:
            pat = {"yolo26x-224": "*cls_yolo26_xlarge_224_best*_preds_val.json", "yolo26n-640": "*cls_yolo26_nano_640_best*_preds_val.json"}[k]
            hits = [os.path.basename(p)[:-len("_preds_val.json")] for p in glob.glob(os.path.join(CLS_DIR, pat)) if "additional" not in p]
            CANDIDATES[k] = sorted(hits)[0] if hits else None
    gt = COCO(GT_VAL)
    frames, first_gt, fps, _ = em.prep(gt, "val")
    by_name = {im["id"]: os.path.basename(im["file_name"]) for im in gt.dataset["images"]}
    positives = {p["image_id"]: True for p in json.load(open(DET_VAL)) if p["score"] >= DET_CONF}
    out = open(OUT_TSV, "w"); out.write("gate\tstem\tthr\trip_alarmed\tnorip_false_episode\tdetector_runs_frac\tvideo_F2\n")
    base = em.evaluate(frames, first_gt, fps, {}, positives, None, DUTY, K, N)
    print(f"no gate: rip {base['rip_alarmed_frac']} false-episode {base['norip_alarmed_frac']} duty {base['detector_runs_frac']}")
    for name, stem in CANDIDATES.items():
        if stem is None: print(f"[{name}] no val dump"); continue
        probs = json.load(open(os.path.join(CLS_DIR, f"{stem}_preds_val.json")))
        cls = {iid: float(probs[b]) for iid, b in by_name.items() if b in probs}
        for thr in THRS:
            m = em.evaluate(frames, first_gt, fps, cls, positives, thr, DUTY, K, N)
            out.write(f"{name}\t{stem}\t{thr}\t{m['rip_alarmed_frac']}\t{m['norip_alarmed_frac']}\t{m['detector_runs_frac']}\t{m['video_F2']}\n")
            print(f"[{name:22s} thr {thr:.1f}] rip {m['rip_alarmed_frac']}  false-episode {m['norip_alarmed_frac']}  duty {m['detector_runs_frac']}  vF2 {m['video_F2']}", flush=True)
    out.close(); print("->", OUT_TSV)


if __name__ == "__main__":
    main()
