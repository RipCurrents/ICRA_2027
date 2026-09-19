#!/usr/bin/env python3
"""Validate the exported RF-DETR detection ONNX against the benchmark (heavy Pareto reference row).

Runs the ONNX (onnxruntime, CPU or CUDA EP) over the val + test sampled frames with rfdetr's own
inference pre/post-processing re-implemented in numpy (square resize to RES, ImageNet mean/std,
sigmoid + top-K over queries x classes, cxcywh -> xyxy scaled to the original size), dumps
benchmark-format COCO results (<stem>_onnx_preds_{val,test}.json) and scores them with the
UNMODIFIED benchmark evaluator on the manual tight box GT (task bbox) exactly like the students
(val-swept conf -> frozen -> test; rows in results/reference_eval_tight.tsv). The row is valid as a
device reference only if it reproduces the camera-ready session's PyTorch row (test F2 0.699 at
conf 0.23) within ~0.005. Optional timed window for latency. Constants below, no CLI args.
Env: conda yoloV26 (onnxruntime present).
"""
import glob
import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
STEM = "1_det_rfdetr_nano_best_best_ema_j3178618"
ONNX = f"/mnt/linux/icra_edge/exports/rfdetr_det/{STEM}/inference_model.onnx"
RES = 384
TOPK = 300
PROVIDERS = ["CPUExecutionProvider"]     # ["CUDAExecutionProvider"] on the 4090 when free
THREADS = 8
SPLITS = ("val", "test")
OUT_DIR = os.path.join(HERE, "predictions", "reference")
EVAL_TSV = os.path.join(HERE, "results", "reference_eval_tight.tsv")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def preprocess(img_bgr):
    h, w = img_bgr.shape[:2]
    x = cv2.resize(img_bgr, (RES, RES), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), w, h


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def decode(dets, logits, w, h, rip_index, cat_id):
    prob = sigmoid(logits[0])                       # (Q, C)
    flat = prob.reshape(-1)
    k = min(TOPK, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    q, c = idx // prob.shape[1], idx % prob.shape[1]
    keep = c == rip_index
    q, s = q[keep], flat[idx][keep]
    b = dets[0][q]                                  # cxcywh normalised
    x1 = (b[:, 0] - b[:, 2] / 2) * w
    y1 = (b[:, 1] - b[:, 3] / 2) * h
    bw, bh = b[:, 2] * w, b[:, 3] * h
    return [dict(category_id=cat_id, bbox=[float(a), float(bb), float(c_), float(d)], score=float(sc))
            for a, bb, c_, d, sc in zip(x1, y1, bw, bh, s)]


def main():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    sess = ort.InferenceSession(ONNX, so, providers=PROVIDERS)
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    print("providers", sess.get_providers(), "outputs", [(o.name, o.shape) for o in sess.get_outputs()], flush=True)
    ED = load_module("ed_ref", os.path.join(REPO_ROOT, "eval", "evaluate_detection.py"))
    ED.SAVE_SWEEP = False
    base = os.path.join(DATASET_ROOT, "labels", "bbox", "coco")
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in SPLITS:
        dst = os.path.join(OUT_DIR, f"{STEM}_onnx_preds_{split}.json")
        if os.path.isfile(dst):
            print(f"{split}: dump exists", flush=True)
            continue
        gt = json.load(open(os.path.join(base, f"{split}.json")))
        cats = gt["categories"]
        cat_id = cats[0]["id"]
        n_cls = sess.get_outputs()[[i for i, o in enumerate(sess.get_outputs()) if o.name == "labels"][0]].shape[-1]
        rip_index = 0   # rfdetr's logit index = position in model.class_names (['rip_current'] -> 0), NOT the COCO id;
                        # verified 2026-08-27 against the PyTorch forward (ONNX == torch outputs; index 1 gave F2 0.01)
        print(f"{split}: {len(gt['images'])} images, categories {cats}, logits classes {n_cls}, rip index {rip_index}", flush=True)
        preds, t0, times = [], time.time(), []
        for i, im in enumerate(gt["images"]):
            p = os.path.join(DATASET_ROOT, "images", im["file_name"].replace("./images/", "").lstrip("./"))
            if not os.path.isfile(p):
                p = os.path.join(DATASET_ROOT, im["file_name"].lstrip("./"))
            img = cv2.imread(p)
            assert img is not None, p
            x, w, h = preprocess(img)
            t1 = time.perf_counter()
            res = dict(zip(outs, sess.run(None, {inp: x})))
            times.append(time.perf_counter() - t1)
            for d in decode(res["dets"], res["labels"], w, h, rip_index, cat_id):
                d["image_id"] = im["id"]
                preds.append(d)
            if i % 1000 == 0:
                print(f"  {split} {i}/{len(gt['images'])} ({time.time() - t0:.0f}s, {np.median(times) * 1000:.1f} ms/img median)", flush=True)
        json.dump(preds, open(dst + ".tmp", "w"))
        os.replace(dst + ".tmp", dst)
        print(f"{split}: {len(preds)} dets in {time.time() - t0:.0f}s", flush=True)
    ED.TASK, ED.MODE, ED.OUT_TSV = "bbox", "bbox", EVAL_TSV
    ED.GT_VAL, ED.GT_TEST = os.path.join(base, "val.json"), os.path.join(base, "test.json")
    gt_v, gt_t = ED.COCO(ED.GT_VAL), ED.COCO(ED.GT_TEST)
    ED.evaluate_pair(gt_v, gt_t, os.path.join(OUT_DIR, f"{STEM}_onnx_preds_val.json"),
                     os.path.join(OUT_DIR, f"{STEM}_onnx_preds_test.json"), STEM + "_onnx")
    print("evaluated ->", EVAL_TSV, flush=True)


if __name__ == "__main__":
    main()
