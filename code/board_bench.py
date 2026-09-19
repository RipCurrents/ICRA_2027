#!/usr/bin/env python3
"""Board-agnostic CPU/NPU benchmark + prediction dump for the edge ladder (RPi 4/5, i.MX95 CPU
fallback, any onnxruntime host). One ONNX file, batch 1: (1) timed window (N_WARMUP + N_TIMED
frames x REPEATS, pre / infer / post split) -> results/board_latency.tsv with board/provider/threads
provenance; (2) optional COCO-results dump over the FULL test split (DUMP_PREDICTIONS) so the
board's outputs are scored by eval/evaluate_detection.py exactly like the 4090 rows (val-frozen conf).
Post-processing = ultralytics-style decode of the exported head (boxes xyxy + scores, class-agnostic
NMS) so the numbers do not depend on ultralytics being installable on the board.
Copy this file + the ONNX + the test image list to the board; needs only numpy, opencv-python,
onnxruntime. Edit the CONFIG block; no CLI args."""
import json
import os
import platform
import time
from datetime import datetime

import numpy as np

# --- CONFIG -------------------------------------------------------------------
ONNX_PATH = "/mnt/linux/icra_edge/exports/t1_nano_k025_t024_s1_n_s1_best_e2_384.onnx"
IMGSZ = 384
BOARD = platform.node()                  # e.g. "rpi5", set explicitly on the board
PROVIDERS = ["CPUExecutionProvider"]     # RPi: CPU; i.MX95: ["VsiNpuExecutionProvider"/"NnapiExecutionProvider"] if eIQ ORT
THREADS = 4                              # intra-op threads (RPi 5: 4 cores)
TASK = "detect"                          # "detect" | "classify" (the cls gate export outputs (1, nc) probs - no boxes/NMS)
COOL_BELOW_C = 65.0                      # wait before each timed window until SoC temp is below this (RPi 4 throttles ~80C
                                         # -> 2026-09-02 ladder invalidated); 0 disables the gate
COOL_TIMEOUT_S = 900                     # give up waiting after this long and proceed (recorded in the row)
IMAGE_ROOT = "/home/user/RipBench/RipBench_v1.2.0/images"   # on the board: the copied test images
TEST_JSON = "/home/user/RipBench/RipBench_v1.2.0/labels/bbox_from_instance/coco/test.json"
N_WARMUP, N_TIMED, REPEATS = 50, 300, 3
DUMP_PREDICTIONS = False                 # True on the board: full test dump for the accuracy row
CONF_FLOOR, IOU_NMS, MAX_DET = 0.001, 0.7, 300
OUT_DIR = "/mnt/linux/icra_edge/board_runs"
LAT_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "board_latency.tsv")
# ------------------------------------------------------------------------------


def letterbox(img, sz):
    import cv2
    h, w = img.shape[:2]
    r = sz / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((sz, sz, 3), 114, np.uint8)
    top, left = (sz - nh) // 2, (sz - nw) // 2
    out[top:top + nh, left:left + nw] = im
    x = out[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, top, left


def classify_preprocess(img, sz):
    """ultralytics CLASSIFICATION transform (= training/validation): RGB, short side resized to sz, center crop
    sz x sz, /255. Deployed gate path v2 (2026-09-07): the letterbox path (v1) disagreed with the offline gate on 16 %
    of val frames and cut rip-frame recall 0.87 -> 0.63; center-crop restores 96 % agreement."""
    import cv2
    h, w = img.shape[:2]
    r = sz / min(h, w)
    im = cv2.resize(img, (max(sz, int(round(w * r))), max(sz, int(round(h * r)))), interpolation=cv2.INTER_LINEAR)
    h2, w2 = im.shape[:2]
    y0, x0 = (h2 - sz) // 2, (w2 - sz) // 2
    x = im[y0:y0 + sz, x0:x0 + sz][:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), 1.0, 0, 0


def read_temp():
    """Max SoC temperature over the thermal zones (deg C); NaN when unreadable."""
    import glob
    vals = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            vals.append(int(open(z).read().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(vals) if vals else float("nan")


def throttled_flags():
    """RPi vcgencmd throttled bits as text; '-' elsewhere."""
    import subprocess
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout.strip().split("=")[-1]
    except Exception:
        return "-"


def wait_cool():
    if not COOL_BELOW_C:
        return read_temp()
    t0 = time.time()
    t = read_temp()
    while t == t and t > COOL_BELOW_C and time.time() - t0 < COOL_TIMEOUT_S:
        print(f"  cooling: {t:.1f}C > {COOL_BELOW_C}C, waiting...", flush=True)
        time.sleep(20)
        t = read_temp()
    return t


def nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (a + b - inter + 1e-9) < thr]
    return keep


def decode(out, r, top, left, conf_floor):
    """ultralytics detect export: (1, 4+nc, N) [cx, cy, w, h, scores...] or end-to-end (1, N, 6) [x1,y1,x2,y2,score,cls]."""
    o = out[0]
    if o.ndim == 3 and o.shape[2] == 6:            # end-to-end (NMS inside the graph)
        d = o[0]
        keep = d[:, 4] >= conf_floor
        boxes, scores = d[keep, :4], d[keep, 4]
    else:
        p = o[0].T if o.shape[1] < o.shape[2] else o[0]  # (N, 4+nc)
        scores = p[:, 4:].max(1)
        keep = scores >= conf_floor
        p, scores = p[keep], scores[keep]
        boxes = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
        if len(boxes):
            k = nms(boxes, scores, IOU_NMS)[:MAX_DET]
            boxes, scores = boxes[k], scores[k]
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
    return boxes, scores


def main():
    import cv2
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    sess = ort.InferenceSession(ONNX_PATH, so, providers=PROVIDERS)
    name = sess.get_inputs()[0].name
    print(f"board {BOARD} | {os.path.basename(ONNX_PATH)} | providers {sess.get_providers()} | threads {THREADS}")
    gt = json.load(open(TEST_JSON))
    items = [(im["id"], os.path.join(IMAGE_ROOT, im["file_name"])) for im in gt["images"]]
    # timed window
    temp_start = wait_cool()
    sample = [cv2.imread(p) for _, p in items[:N_WARMUP + N_TIMED]]
    prep = classify_preprocess if TASK == "classify" else letterbox
    pre = [prep(im, IMGSZ) for im in sample]
    for x, *_ in pre[:N_WARMUP]:
        sess.run(None, {name: x})
    infer, pre_t, post_t = [], [], []
    for rep in range(REPEATS):
        for im, (x, r, top, left) in zip(sample[N_WARMUP:], pre[N_WARMUP:]):
            t0 = time.perf_counter(); xx, *_ = prep(im, IMGSZ); t1 = time.perf_counter()
            out = sess.run(None, {name: xx}); t2 = time.perf_counter()
            _ = float(out[0][0].max()) if TASK == "classify" else decode(out, r, top, left, 0.25)
            t3 = time.perf_counter()
            pre_t.append((t1 - t0) * 1e3); infer.append((t2 - t1) * 1e3); post_t.append((t3 - t2) * 1e3)
    m = np.mean(infer); sd = np.std(np.array(infer).reshape(REPEATS, -1).mean(1))
    os.makedirs(os.path.dirname(LAT_TSV), exist_ok=True)
    hdr = not os.path.isfile(LAT_TSV)
    with open(LAT_TSV, "a") as f:
        if hdr:
            f.write("timestamp\tboard\tmodel\timgsz\tprovider\tthreads\tn_timed\trepeats\tpre_ms\tinfer_ms\tinfer_std\tpost_ms\tfps_total\ttask\ttemp_start_C\ttemp_end_C\tthrottled\n")
        f.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{BOARD}\t{os.path.basename(ONNX_PATH)}\t{IMGSZ}\t{sess.get_providers()[0]}\t{THREADS}\t{N_TIMED}\t{REPEATS}\t"
                f"{np.mean(pre_t):.2f}\t{m:.2f}\t{sd:.2f}\t{np.mean(post_t):.2f}\t{1000 / (np.mean(pre_t) + m + np.mean(post_t)):.1f}\t"
                f"{TASK}\t{temp_start:.1f}\t{read_temp():.1f}\t{throttled_flags()}\n")
    print(f"pre {np.mean(pre_t):.1f} ms | infer {m:.1f} ± {sd:.1f} ms | post {np.mean(post_t):.1f} ms | {1000 / (np.mean(pre_t) + m + np.mean(post_t)):.1f} fps end-to-end")
    if DUMP_PREDICTIONS and TASK == "detect":
        os.makedirs(OUT_DIR, exist_ok=True)
        res = []
        for i, (iid, p) in enumerate(items):
            im = cv2.imread(p); x, r, top, left = letterbox(im, IMGSZ)
            boxes, scores = decode(sess.run(None, {name: x}), r, top, left, CONF_FLOOR)
            for b, s in zip(boxes, scores):
                res.append({"image_id": int(iid), "category_id": 1, "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "score": float(s)})
            if i % 1000 == 0:
                print(f"  dump {i}/{len(items)}", flush=True)
        stem = f"{os.path.splitext(os.path.basename(ONNX_PATH))[0]}_{BOARD}"
        out = os.path.join(OUT_DIR, f"{stem}_preds_test.json")
        json.dump(res, open(out, "w"))
        print(f"-> {out} ({len(res)} preds); evaluate with eval/evaluate_detection.py (val-frozen conf from the 4090 row)")


if __name__ == "__main__":
    main()
