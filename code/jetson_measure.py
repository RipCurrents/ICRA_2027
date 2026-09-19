#!/usr/bin/env python3
"""Jetson Orin Nano measurement driver (execution-only on arrival; written 2026-08-27 without the device).

Per ONNX model x resolution x precision: (1) build a TensorRT engine with trtexec (cached),
(2) time batch-1 inference (N_WARMUP + N_TIMED frames x REPEATS) with the TensorRT Python API,
(3) sample the on-board INA3221 rails at ~10 Hz (tegrastats) during the timed window and at idle
-> W and mJ/frame, (4) optionally dump COCO results over the FULL test split for the benchmark
evaluator (accuracy per precision/resolution), (5) duty-cycle runs at DUTY_FPS for DUTY_SECONDS each
-> average W. Wall-meter readings are typed in by hand into results/device_wall_meter.tsv
(timestamped windows). Rows -> results/device_{latency,energy,accuracy,duty}.tsv with device /
power-mode / jetson_clocks / precision / imgsz / engine-hash provenance.
Copy this file + board_bench.py (decode + letterbox) + the ONNX files + test images to the device.
Constants below; no CLI args. Needs: tensorrt, pycuda (or cuda-python), numpy, opencv, tegrastats.
"""
import hashlib
import json
import os
import platform
import re
import subprocess
import threading
import time
from datetime import datetime

import numpy as np

# --- CONFIG -------------------------------------------------------------------
DEVICE = platform.node()                       # e.g. "orin-nano-8gb"
POWER_MODE = "15W"                             # what `nvpmodel -q` says; set before running
JETSON_CLOCKS = True                           # `sudo jetson_clocks` applied? (latency rows: True; duty rows: False)
ONNX_DIR = "/home/jetson/icra/exports"         # copied from /mnt/linux/icra_edge/exports
MODELS = {                                     # stem -> list of resolutions (ONNX files <stem>_<sz>.onnx)
    "t1_nano_k025_t024_s1_n_s1_best_e2": (640, 512, 384),
    "canon_bfi_n_s1_best_e18": (640, 512, 384),
    "t1_nano_k025_t024_s1_s_s3_best_e3": (640, 384),
    "canon_bfi_s_s1_best_e1": (640, 384),
}
PRECISIONS = ("fp32", "fp16", "int8")
INT8_CALIB_DIR = "/home/jetson/icra/calib_train_500"   # 500 TRAIN images (never val/test)
ENGINE_DIR = "/home/jetson/icra/engines"
IMAGE_ROOT = "/home/jetson/icra/images"        # test images copied from RipBench_v1.2.0/images
TEST_JSON = "/home/jetson/icra/labels/bbox_from_instance/coco/test.json"
N_WARMUP, N_TIMED, REPEATS = 200, 1000, 5
DUMP_PREDICTIONS = True
DUTY_FPS = (5.0, 2.0, 1.0, 0.2)
DUTY_SECONDS = 600
IDLE_SECONDS = 60
RESULTS_DIR = "/home/jetson/icra/results"
TEGRASTATS_INTERVAL_MS = 100
# ------------------------------------------------------------------------------

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_bench import letterbox, decode  # noqa: E402  (same pre/post-processing as the CPU boards)


# ---------- power sampling (tegrastats) ----------
class Tegrastats:
    """Samples `tegrastats --interval` in a thread; parses VDD_IN / VDD_CPU_GPU_CV / VDD_SOC (mW)."""
    RAIL_RE = re.compile(r"(VDD_IN|VDD_CPU_GPU_CV|VDD_SOC)\s+(\d+)mW/(\d+)mW")

    def __init__(self):
        self.samples, self.proc, self.thread = [], None, None

    def _reader(self):
        for line in self.proc.stdout:
            t = time.time()
            rails = {k: int(v) for k, v, _ in self.RAIL_RE.findall(line)}
            if rails:
                self.samples.append((t, rails))

    def start(self):
        self.samples = []
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(TEGRASTATS_INTERVAL_MS)],
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        self.proc.terminate()
        self.thread.join(timeout=2)
        return list(self.samples)

    @staticmethod
    def mean_w(samples, t0, t1, rail="VDD_IN"):
        v = [s[rail] for t, s in samples if t0 <= t <= t1 and rail in s]
        return (np.mean(v) / 1000.0, len(v)) if v else (float("nan"), 0)


# ---------- TensorRT ----------
def build_engine(onnx_path, precision):
    os.makedirs(ENGINE_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    eng = os.path.join(ENGINE_DIR, f"{stem}_{precision}.engine")
    if os.path.isfile(eng):
        return eng
    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={eng}", "--workspace=2048"]
    if precision == "fp16":
        cmd.append("--fp16")
    if precision == "int8":
        cache = os.path.join(ENGINE_DIR, f"{stem}_int8.cache")
        cmd += ["--int8", f"--calib={cache}"]
        if not os.path.isfile(cache):
            # calibration cache from 500 train images via the polygraphy/trtexec calibrator convention:
            # simplest robust path = build once with --int8 and a Python calibrator (see calib_int8.py, to be written on the device)
            raise RuntimeError("INT8 calibration cache missing - run calib_int8.py first")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return eng


class TrtRunner:
    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bufs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.bufs[name] = (host, dev, shape, dtype, self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
            self.ctx.set_tensor_address(name, int(dev))
        self.in_name = [n for n, b in self.bufs.items() if b[4]][0]
        self.out_names = [n for n, b in self.bufs.items() if not b[4]]

    def infer(self, x):
        host, dev, shape, dtype, _ = self.bufs[self.in_name]
        np.copyto(host, x.astype(dtype).ravel())
        self.cuda.memcpy_htod_async(dev, host, self.stream)
        self.ctx.execute_async_v3(stream_handle=self.stream.handle)
        outs = []
        for n in self.out_names:
            h, d, s, dt, _ = self.bufs[n]
            self.cuda.memcpy_dtoh_async(h, d, self.stream)
            outs.append((h, s))
        self.stream.synchronize()
        return [h.reshape(s) for h, s in outs]


def engine_hash(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:10]


def tsv(name, header, row):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    p = os.path.join(RESULTS_DIR, name)
    new = not os.path.isfile(p)
    with open(p, "a") as f:
        if new:
            f.write(header + "\n")
        f.write(row + "\n")


def main():
    import cv2
    gt = json.load(open(TEST_JSON))
    items = [(im["id"], os.path.join(IMAGE_ROOT, im["file_name"])) for im in gt["images"]]
    power = Tegrastats()
    # idle baseline
    power.start(); time.sleep(IDLE_SECONDS); idle = power.stop()
    idle_w, n_idle = Tegrastats.mean_w(idle, idle[0][0], idle[-1][0])
    print(f"idle {idle_w:.2f} W ({n_idle} samples)")
    prov = f"{DEVICE}\t{POWER_MODE}\t{int(JETSON_CLOCKS)}"
    for stem, sizes in MODELS.items():
        for sz in sizes:
            onnx = os.path.join(ONNX_DIR, f"{stem}_{sz}.onnx")
            sample = [cv2.imread(p) for _, p in items[:N_WARMUP + N_TIMED]]
            pre = [letterbox(im, sz) for im in sample]
            for prec in PRECISIONS:
                try:
                    eng = build_engine(onnx, prec)
                except Exception as e:
                    print(f"SKIP {stem} {sz} {prec}: {e}"); continue
                r = TrtRunner(eng); eh = engine_hash(eng)
                for x, *_ in pre[:N_WARMUP]:
                    r.infer(x)
                # timed window with power sampling
                power.start(); t_start = time.time(); lat = []
                for rep in range(REPEATS):
                    for x, *_ in pre[N_WARMUP:]:
                        t0 = time.perf_counter(); r.infer(x); lat.append((time.perf_counter() - t0) * 1e3)
                t_end = time.time(); samples = power.stop()
                per_rep = np.array(lat).reshape(REPEATS, -1).mean(1)
                w, n = Tegrastats.mean_w(samples, t_start, t_end)
                w_cg, _ = Tegrastats.mean_w(samples, t_start, t_end, "VDD_CPU_GPU_CV")
                mj = (w - idle_w) * np.mean(lat)  # W * ms = mJ
                tsv("device_latency.tsv", "timestamp\tdevice\tpower_mode\tjetson_clocks\tmodel\timgsz\tprecision\tengine\tn_timed\trepeats\tinfer_ms\tinfer_std\tfps",
                    f"{datetime.now():%Y-%m-%d %H:%M}\t{prov}\t{stem}\t{sz}\t{prec}\t{eh}\t{N_TIMED}\t{REPEATS}\t{per_rep.mean():.3f}\t{per_rep.std():.3f}\t{1000/per_rep.mean():.1f}")
                tsv("device_energy.tsv", "timestamp\tdevice\tpower_mode\tjetson_clocks\tmodel\timgsz\tprecision\tengine\tidle_W\tactive_W_VDD_IN\tactive_W_CPU_GPU_CV\tn_samples\tmJ_per_frame\twindow_start\twindow_end",
                    f"{datetime.now():%Y-%m-%d %H:%M}\t{prov}\t{stem}\t{sz}\t{prec}\t{eh}\t{idle_w:.2f}\t{w:.2f}\t{w_cg:.2f}\t{n}\t{mj:.2f}\t{t_start:.0f}\t{t_end:.0f}")
                print(f"{stem} {prec} @{sz}: {per_rep.mean():.2f} ± {per_rep.std():.2f} ms, {w:.2f} W (idle {idle_w:.2f}), {mj:.1f} mJ/frame")
                if DUMP_PREDICTIONS:
                    res = []
                    for iid, p in items:
                        im = cv2.imread(p); x, rr, top, left = letterbox(im, sz)
                        boxes, scores = decode(r.infer(x), rr, top, left, 0.001)
                        for b, s in zip(boxes, scores):
                            res.append({"image_id": int(iid), "category_id": 1, "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])], "score": float(s)})
                    out = os.path.join(RESULTS_DIR, "predictions", f"{stem}_{sz}_{prec}_{DEVICE}_preds_test.json")
                    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"))
                    print(f"  dumped {len(res)} preds -> {out}")
                # duty-cycle windows (detector alone; cascade rows come from cascade_device.py)
                for fps in DUTY_FPS:
                    period = 1.0 / fps; power.start(); t0 = time.time(); k = 0
                    while time.time() - t0 < DUTY_SECONDS:
                        t = time.perf_counter(); r.infer(pre[N_WARMUP + (k % N_TIMED)][0]); k += 1
                        time.sleep(max(0.0, period - (time.perf_counter() - t)))
                    t1 = time.time(); samples = power.stop(); w, n = Tegrastats.mean_w(samples, t0, t1)
                    tsv("device_duty.tsv", "timestamp\tdevice\tpower_mode\tjetson_clocks\tmodel\timgsz\tprecision\tduty_fps\tseconds\tframes\tavg_W\tidle_W\tmJ_per_frame_total",
                        f"{datetime.now():%Y-%m-%d %H:%M}\t{prov}\t{stem}\t{sz}\t{prec}\t{fps}\t{DUTY_SECONDS}\t{k}\t{w:.2f}\t{idle_w:.2f}\t{(w*DUTY_SECONDS*1000)/max(k,1):.1f}")
                    print(f"  duty {fps} fps: {w:.2f} W over {k} frames")
    print("DEVICE_MEASURE_DONE")


if __name__ == "__main__":
    main()
