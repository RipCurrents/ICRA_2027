#!/usr/bin/env python3
"""Inference-time temporal smoothing of STUDENT outputs (ICRA edge pipeline).

The students run frame by frame; a beach-camera stream gives free temporal context. For
every dump <stem>_preds_{val,test}.json (benchmark predictor format, conf >= 0.001) this
applies the SAME IoU-linked window smoothing used for the teacher labels
(build_teacher_view.smooth_video: score = mean over the +-k window, unmatched frames count 0,
box = score-weighted mean) along each video's SAMPLED-frame sequence (every 6th native frame,
i.e. k=1 ~ +-0.2 s, k=2 ~ +-0.4 s at 30 fps), re-applies NMS (the dumps are post-NMS; smoothed duplicates must be re-suppressed),
writes <stem>_sm<k>_preds_{val,test}.json next to the originals, and scores them with the benchmark evaluator exactly like train_student.py
(val-swept conf -> frozen -> test; both GT views; rows appended to
results/student_eval_{bfi,tight}.tsv with the _sm<k> stem). Constants below, no CLI args.
Env: conda yoloV26 (CPU only).
"""
import glob
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_teacher_view import iou_matrix, nms, MATCH_IOU  # noqa: E402  (same linking rule as the labels)

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
PRED_DIRS = [os.path.join(HERE, "predictions", "students"), "/mnt/linux/icra_edge/cluster_pull/predictions",
             os.path.join(DATASET_ROOT, "models", "bbox_from_instance", "best_models", "predictions")]   # benchmark dumps (read-only)
OUT_DIR = os.path.join(HERE, "predictions", "students")   # smoothed dumps (symlink -> /mnt/linux)
STEM_RX = [r"^1_bfi_yolo26_nano_best_e13_j2853799$"]   # 2026-08-29 cross-application (the internal critic 2x2): the BENCHMARK yolo26n bfi dump
                                                     # (AutoBatch recipe, test F2 0.444) through this smoother; val-frozen, ONE test read.
# earlier passes: [r"^canon_bfi_n_s\d+_best_e\d+$", r"^t1_nano_k025_t024_s1_n_s\d+_best_e\d+$", r"^t1_nano_k025_t024_s1_s_s\d+_best_e\d+$"]
K_FRAMES = [1, 2]            # +-k sampled frames (CAUSAL=False) or the past 2k frames (CAUSAL=True)
CAUSAL = True                # 2026-08-29: causal window [t-2k, t] (no output delay; deployable inside the duty-cycled pipeline) - suffix _csm<k>;
NEIGHBOUR_MIN_SCORE = 0.01   # neighbour candidates below the sweep floor cannot change any swept operating point
ZERO_FILL = True             # window frames without a linked box count as score 0 (an isolated one-frame box is down-weighted = the
                             # persistence prior); False = mean over MATCHED frames only (matched-frames-only rule) - mechanism check 2026-08-29
RE_NMS = True                # False = skip the re-NMS after smoothing (mechanism check)
NMS_IOU = 0.7                # re-suppress after smoothing (ultralytics' predict NMS IoU): without it, sub-threshold
                             # duplicates of a strong box inherit its score from the neighbours and become FPs (2026-08-27)
EVAL_TASKS = ("bbox_from_instance", "bbox")
EVAL_TSV = {"bbox_from_instance": os.path.join(HERE, "results", "student_eval_bfi.tsv"),
            "bbox": os.path.join(HERE, "results", "student_eval_tight.tsv")}
VIDEO_RE = re.compile(r"(RipBench-(?:NR-)?\d+)_(\d+)\.jpg$")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def tsv_has(tsv, stem, split):
    if not os.path.isfile(tsv):
        return False
    return any(l.split("\t")[1] == stem and l.split("\t")[2] == split for l in open(tsv) if "\t" in l) if False else \
        any((stem + "\t") in l and ("\t" + split + "\t") in l for l in open(tsv))


def smooth_dump(preds, gt_json, k):
    gt = json.load(open(gt_json))
    order = {}
    for im in gt["images"]:
        v, idx = VIDEO_RE.search(im["file_name"]).groups()
        order.setdefault(v, []).append((int(idx), im["id"]))
    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)
    out = []
    for v, frames in order.items():
        frames.sort()
        ids = [i for _, i in frames]
        arr = {}
        for pos, iid in enumerate(ids):
            ps = by_img.get(iid, [])
            if ps:
                b = np.array([p["bbox"] for p in ps], dtype=np.float64)
                b[:, 2:] += b[:, :2]                       # xywh -> xyxy
                arr[pos] = (np.array([p["score"] for p in ps]), b, ps)
        for pos, (s_t, b_t, ps) in arr.items():
            lo, hi = (max(0, pos - 2 * k), pos) if CAUSAL else (max(0, pos - k), min(len(ids) - 1, pos + k))
            acc_s, acc_b, acc_w = s_t.copy(), b_t * s_t[:, None], s_t.copy()
            n_matched = np.ones(len(s_t))
            for u in range(lo, hi + 1):
                if u == pos or u not in arr:
                    continue
                s_u, b_u, _ = arr[u]
                keep = s_u >= NEIGHBOUR_MIN_SCORE
                if not keep.any():
                    continue
                s_u, b_u = s_u[keep], b_u[keep]
                ious = iou_matrix(b_t, b_u)
                j = ious.argmax(1)
                ok = ious[np.arange(len(b_t)), j] >= MATCH_IOU
                m = np.where(ok, s_u[j], 0.0)
                acc_s += m
                acc_b += b_u[j] * m[:, None]
                acc_w += m
                n_matched += ok
            s_sm = acc_s / (hi - lo + 1) if ZERO_FILL else acc_s / n_matched
            b_sm = acc_b / np.maximum(acc_w, 1e-9)[:, None]
            keep = nms(b_sm, s_sm, NMS_IOU) if RE_NMS else np.arange(len(s_sm))
            for p, s, bb in ((ps[i], s_sm[i], b_sm[i]) for i in keep):
                out.append({"image_id": p["image_id"], "category_id": p["category_id"],
                            "bbox": [float(bb[0]), float(bb[1]), float(bb[2] - bb[0]), float(bb[3] - bb[1])],
                            "score": float(s)})
    return out


def main():
    ED = load_module("ed_sm", os.path.join(REPO_ROOT, "eval", "evaluate_detection.py"))
    ED.SAVE_SWEEP = False
    stems = []
    for d in PRED_DIRS:
        for p in glob.glob(os.path.join(d, "*_preds_test.json")):
            stem = os.path.basename(p)[:-len("_preds_test.json")]
            if any(re.match(rx, stem) for rx in STEM_RX) and os.path.isfile(os.path.join(d, f"{stem}_preds_val.json")):
                stems.append((stem, d))
    stems = sorted(set(stems))
    print(f"{len(stems)} stems x k in {K_FRAMES}", flush=True)
    gts = {t: {s: os.path.join(DATASET_ROOT, "labels", t, "coco", f"{s}.json") for s in ("val", "test")} for t in EVAL_TASKS}
    for stem, d in stems:
        for k in K_FRAMES:
            new = f"{stem}_{'csm' if CAUSAL else 'sm'}{k}"
            for split in ("val", "test"):
                dst = os.path.join(OUT_DIR, f"{new}_preds_{split}.json")
                if os.path.isfile(dst):
                    continue
                t0 = time.time()
                preds = json.load(open(os.path.join(d, f"{stem}_preds_{split}.json")))
                sm = smooth_dump(preds, gts["bbox_from_instance"][split], k)   # frame order is view-independent
                json.dump(sm, open(dst + ".tmp", "w"))
                os.replace(dst + ".tmp", dst)
                print(f"[{new}] {split}: {len(preds)} -> {len(sm)} dets in {time.time() - t0:.0f}s", flush=True)
            for task in EVAL_TASKS:
                if tsv_has(EVAL_TSV[task], new, "test"):
                    continue
                ED.TASK, ED.MODE, ED.OUT_TSV = task, "bbox", EVAL_TSV[task]
                ED.GT_VAL, ED.GT_TEST = gts[task]["val"], gts[task]["test"]
                gt_v, gt_t = ED.COCO(ED.GT_VAL), ED.COCO(ED.GT_TEST)
                ED.evaluate_pair(gt_v, gt_t, os.path.join(OUT_DIR, f"{new}_preds_val.json"),
                                 os.path.join(OUT_DIR, f"{new}_preds_test.json"), new)
                print(f"[{new}] evaluated on {task}", flush=True)


if __name__ == "__main__":
    main()
