#!/usr/bin/env python3
"""Scenario S2 (the author 2026-09-07): CLASSIFICATION-ONLY alarm (lifeguard alert: "a rip is present", no localisation).
VAL ONLY: for each candidate classifier's val probabilities (benchmark = training transform), alarm at 1 Hz when
P(rip) >= t (k=n=1, GAP_S episode merge, no detector): rip videos alarmed, no-rip false-episode clips, time-to-alarm,
video-F2 over a coarse grid; selection rule = highest val video-F2, ties -> higher t. Output
results/classify_only_alarm_val.tsv. Constants at top, no CLI args."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
GT_VAL = os.path.join(DATASET_ROOT, "labels", "bbox_from_instance", "coco", "val.json")
CLS_DIR = os.path.join(DATASET_ROOT, "models", "classification", "best_models", "predictions")
CANDIDATES = {"yolo26n-224 (deployed gate)": "3_cls_yolo26_nano_224_best_e50_j2856380",
              "yolo26m-224": "11_cls_yolo26_medium_224_best_e39_j2856390",
              "yolo26x-224": None, "resnet34-640": "7_cls_resnet_resnet34_640_best_e19_j2857467"}
THRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DUTY, K, N = 1.0, 1, 1
OUT_TSV = os.path.join(HERE, "results", "classify_only_alarm_val.tsv")


def main():
    import event_metrics as em
    from pycocotools.coco import COCO
    hits = [os.path.basename(p)[:-len("_preds_val.json")] for p in glob.glob(os.path.join(CLS_DIR, "*cls_yolo26_xlarge_224_best*_preds_val.json")) if "additional" not in p]
    CANDIDATES["yolo26x-224"] = sorted(hits)[0] if hits else None
    gt = COCO(GT_VAL); frames, first_gt, fps, _ = em.prep(gt, "val")
    by_name = {im["id"]: os.path.basename(im["file_name"]) for im in gt.dataset["images"]}
    out = open(OUT_TSV, "w"); out.write("classifier\tthr\trip_alarmed\tlatency_median_s\tlatency_p90_s\tnorip_false_episode\tvideo_F2\tselected\n")
    for name, stem in CANDIDATES.items():
        if stem is None: continue
        probs = json.load(open(os.path.join(CLS_DIR, f"{stem}_preds_val.json")))
        cls = {iid: float(probs[b]) for iid, b in by_name.items() if b in probs}
        rows = []
        for t in THRS:
            positives = {iid: True for iid, pr in cls.items() if pr >= t}   # the classifier IS the alarm
            m = em.evaluate(frames, first_gt, fps, {}, positives, None, DUTY, K, N)
            rows.append((t, m))
        best = max(rows, key=lambda r: (float(r[1]["video_F2"]), r[0]))[0]
        for t, m in rows:
            out.write(f"{name}\t{t}\t{m['rip_alarmed_frac']}\t{m['latency_median_s']}\t{m['latency_p90_s']}\t{m['norip_alarmed_frac']}\t{m['video_F2']}\t{'*' if t == best else ''}\n")
        sel = next(m for t, m in rows if t == best)
        print(f"[{name:28s}] val-selected thr {best:.1f}: rip {sel['rip_alarmed_frac']} false-episode {sel['norip_alarmed_frac']} TTA {sel['latency_median_s']}/{sel['latency_p90_s']} s vF2 {sel['video_F2']}", flush=True)
    out.close(); print("->", OUT_TSV)


if __name__ == "__main__":
    main()
