#!/usr/bin/env python3
"""Rip-current edge app — deployment / visualisation companion to the ICRA 2027 edge paper.

Runs the paper's edge pipeline (yolo26n-cls 224 GATE -> light yolo26 DETECTOR at a duty cycle) on a video file,
an RTSP/HTTP stream or a webcam, shows a live overlay (boxes + scores, gate state, detector duty, processing fps
and, on a Jetson, the board power from tegrastats) and exports a report (annotated mp4 + per-frame TSV + summary
json) into OUT_ROOT/<timestamp>_<source>_<model>_<size>/.  Two modes over the SAME pipeline:
  * UI (default):  python app.py                         -> Gradio at http://<host>:7860
  * headless:      RIPAPP_MODE=headless python app.py    -> runs HEADLESS_JOBS (or the json list in $RIPAPP_JOBS)
Inference path = code/board_bench.py (letterbox + ONNX decode, validated against the benchmark
evaluator within +-0.005 F2) and hil_demo.py's gate handling (copied verbatim); TensorRT engines built by
jetson_measure.py / trtexec are used when present (backend "auto"/"trt-engine").  Nothing here is a research
tool: thresholds are NEVER tuned in the app — the default conf per model/size is the VAL-frozen value from
results/student_eval_bfi.tsv (column selected_conf) and the gate threshold is the val-selected 0.50.
Constants below (house rules: no CLI args).  The RIPAPP_* environment overrides exist only so run.sh can switch
the mode / paths on another machine (Jetson) without editing this file.
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime

import numpy as np

APP_VERSION = "0.1 (2026-08-27)"

# --- CONFIG -------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))                  # <release>/app
REPO_DIR = os.path.dirname(HERE)
EDGE_DIR = os.environ.get("RIPAPP_EDGE_DIR", os.path.join(REPO_DIR, "code"))   # board_bench, hil_demo, jetson_measure, results/
EXPORTS_DIR = os.environ.get("RIPAPP_EXPORTS_DIR", "/mnt/linux/icra_edge/exports")
MANIFEST_TSV = os.environ.get("RIPAPP_MANIFEST", os.path.join(REPO_DIR, "results", "export_manifest.tsv"))
EVAL_TSV = os.environ.get("RIPAPP_EVAL_TSV", os.path.join(REPO_DIR, "results", "student_eval_bfi.tsv"))
GATE_ONNX = os.path.join(EXPORTS_DIR, "yolo26n_cls_224_gate.onnx")
GATE_IMGSZ = 224
GATE_THR = 0.50                        # val-selected gate threshold (RESULTS.md, cascade section)
DEFAULT_MODEL = "t1_nano_k025_t024_s1_n_s1_best_e2"   # the paper's dense-supervision yolo26n, seed 1
DEFAULT_IMGSZ = 384
DEFAULT_DUTY_FPS = 1.0                 # decisions per second of video time (rips evolve over seconds–minutes)
DEFAULT_BACKEND = "auto"               # auto = TensorRT engine if one exists for the model, else onnxruntime CPU | cpu | cuda | trt-engine
THREADS = 4                            # onnxruntime intra-op threads (an edge-CPU-like budget on any host)
NMS_IOU = 0.7                          # class-agnostic NMS on the decoded boxes (= board_bench.IOU_NMS; the benchmark dumps are post-NMS 0.7 and the
                                       # end-to-end export emits near-duplicate low-score boxes at the recall-heavy frozen conf); None = raw boxes
ENGINE_DIR = os.environ.get("RIPAPP_ENGINE_DIR", "/home/jetson/icra/engines")   # jetson_measure.build_engine: <stem>_<precision>.engine
ENGINE_PRECISION = "fp16"
OUT_ROOT = os.environ.get("RIPAPP_OUT_ROOT", "/mnt/linux/icra_edge/app_runs")
DISPLAY_MAX_WIDTH = 1280               # frames sent to the browser are resized to this width
DISPLAY_FPS = 8.0                      # max UI refresh rate (the report mp4 keeps every processed frame)
LIVE_LOOP_FPS = 10.0                   # webcam / stream: processing + recording rate between ticks (the mp4 fps for live sources)
HUD_SCALE = 1.6                        # overlay text size (cv2 font scale at 1080p): reviewers watch the demo at 720p in a browser
POWER_IDLE_SECONDS = 3.0               # Jetson: idle window (models loaded, nothing running) sampled before the run, then ...
POWER_BURST_SECONDS = 2.0              # ... a continuous-inference window -> idle_W / active_W; mJ per decision = (active - idle) x t_decision
RIP_WINDOWS = {                        # demo clips: seconds of video containing a rip (None = to the end) -> false-alarm count of the
    "deployment_replay": (20.0, 50.0), # decisions (NR-021 20 s + RipBench-038 30 s + NR-021 20 s); all-rip test videos: whole clip;
    "RipBench-038": (0.0, None), "RipBench-025": (0.0, None), "RipBench-125": (0.0, None),
}                                      # headless jobs: key "rip_window": [start, end]; other sources: false alarms not counted
RECORD_MAX_WIDTH = None                # None = native resolution in the report mp4 (set e.g. 1280 on a slow board)
TRANSCODE_H264 = True                  # re-encode the mp4v report with ffmpeg (libx264) so browsers can play it; needs ffmpeg on PATH
SAMPLE_CLIPS = OrderedDict([           # one-click demo material (held-out TEST videos; see ICRA_2027/notes/device_runbook.md)
    ("RipBench-038 — fixed beach camera (test)", "/home/user/RipBench/Videos/Rips/RipBench-038.mp4"),
    ("RipBench-025 — static drone (test)", "/home/user/RipBench/Videos/Rips/RipBench-025.mp4"),
    ("RipBench-125 — documented failure case (test)", "/home/user/RipBench/Videos/Rips/RipBench-125.mp4"),
    ("Deployment replay — no-rip / rip / no-rip (NR-021 + 038)", "/mnt/linux/icra_edge/hil/deployment_replay.mp4"),
])
CLIPS_DIR = os.environ.get("RIPAPP_CLIPS_DIR")   # another machine (Jetson bundle): the sample clips copied into one folder, matched by basename
if CLIPS_DIR:
    SAMPLE_CLIPS = OrderedDict((k, p if os.path.isfile(p) else v) for k, v in SAMPLE_CLIPS.items()
                               for p in [os.path.join(CLIPS_DIR, os.path.basename(v))])
HEAVY_MODELS = OrderedDict([           # heavy Pareto references validated by the ICRA session; own pre/post (train/edge_distill/rfdetr_onnx_bench.py,
    ("rfdetr_nano_det", dict(           # imported, never modified): square resize 384 + ImageNet norm, sigmoid top-300, DETR -> no NMS; 384 px only
        label="RF-DETR-n · heavy reference (30 M params)", family="rfdetr", imgsz=384, rip_index=0,
        onnx=os.path.join(EXPORTS_DIR, "rfdetr_det", "1_det_rfdetr_nano_best_best_ema_j3178618", "inference_model.onnx"),
        conf=0.26, test_f2=0.706, gt_view="tight", params_M=30.0, gflops=None)),   # val-frozen conf; test F2 on the manual TIGHT GT (students: bfi)
])
SERVER_NAME, SERVER_PORT = "0.0.0.0", int(os.environ.get("RIPAPP_PORT", "7860"))
MODE = os.environ.get("RIPAPP_MODE", "ui")          # "ui" | "headless"
HEADLESS_DEFAULTS = dict(source_kind="file", model=DEFAULT_MODEL, imgsz=DEFAULT_IMGSZ, backend=DEFAULT_BACKEND,
                         duty_fps=DEFAULT_DUTY_FPS, gate=True, gate_thr=GATE_THR, conf=None, record=True,
                         pace_realtime=False, max_seconds=0, out_name=None)
HEADLESS_JOBS = [                      # batch demo videos for the ICRA session (each key overrides HEADLESS_DEFAULTS)
    {"source": SAMPLE_CLIPS["RipBench-038 — fixed beach camera (test)"]},
    {"source": SAMPLE_CLIPS["RipBench-025 — static drone (test)"]},
    {"source": SAMPLE_CLIPS["RipBench-125 — documented failure case (test)"]},
    {"source": SAMPLE_CLIPS["Deployment replay — no-rip / rip / no-rip (NR-021 + 038)"]},
]
# ------------------------------------------------------------------------------
RTSP_TRANSPORT = None                  # None = let ffmpeg negotiate (UDP, TCP fallback); "tcp" forces TCP-interleaved (lossy WiFi links,
                                       # cameras behind NAT) — but some servers only speak UDP (VLC's RTSP output answers 461 to TCP)
import cv2  # noqa: E402

sys.path.insert(0, EDGE_DIR)
from board_bench import letterbox, decode, nms, classify_preprocess  # noqa: E402  (the validated pre/post-processing; never modified here)

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED, GREEN, GREY, WHITE, BLACK, AMBER = (40, 40, 230), (80, 200, 80), (170, 170, 170), (255, 255, 255), (0, 0, 0), (40, 170, 240)
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".ts")
STOP_EVENT = threading.Event()
RUN_LOCK = threading.Lock()


# ---------- model registry (export manifest x val-frozen operating points) ----------
def _read_tsv(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    if not lines:
        return []
    hdr = lines[0].split("\t")
    rows = []
    for l in lines[1:]:
        parts = l.split("\t")
        if parts == hdr:            # repeated header lines occur in the appended eval TSVs
            continue
        rows.append(dict(zip(hdr, parts)))
    return rows


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


_NAMES = [  # (regex on the checkpoint stem, label with the seed, rank in the dropdown)
    (r"^t1_nano_k025_t024_s1_n_s(\d)_best", "yolo26n · dense supervision (paper student) · seed {0}", 0),
    (r"^t1_nano_k025_t024_s1_s_s(\d)_best", "yolo26s · dense supervision · seed {0}", 1),
    (r"^canon_bfi_n_s(\d)_best", "yolo26n · canonical expert labels · seed {0}", 2),
    (r"^canon_bfi_s_s(\d)_best", "yolo26s · canonical expert labels · seed {0}", 3),
    (r"^interp_gt_s1_n_s(\d)_best", "yolo26n · interpolated expert labels (teacher-free) · seed {0}", 4),
    (r"^kd_bfi_x_n_s(\d)_best", "yolo26n · native KD from yolo26x · seed {0}", 5),
    (r"^kd_t1_n_s(\d)_best", "yolo26n · dense + native KD · seed {0}", 6),
    (r"^t1_nano_k0_t024_s1_n_s(\d)_best", "yolo26n · dense, no temporal smoothing · seed {0}", 7),
    (r"^t1_nano_k025_t015_s1_n_s(\d)_best", "yolo26n · dense, tau 0.15 · seed {0}", 8),
    (r"^t1_nano_k025_t035_s1_n_s(\d)_best", "yolo26n · dense, tau 0.35 · seed {0}", 8),
    (r"^t1_nano_k025_t024_s2_n_s(\d)_best", "yolo26n · dense, stride 2 (half density) · seed {0}", 9),
    (r"^t1_nano_k025_t024_s4_n_s(\d)_best", "yolo26n · dense, stride 4 (quarter density) · seed {0}", 9),
    (r"^t1_nano_k025_t024_s1_keepempty_n_s(\d)_best", "yolo26n · dense, keep empty frames · seed {0}", 10),
    (r"^t1_nano_k025_t024_s1_noneg_n_s(\d)_best", "yolo26n · dense, no no-rip negatives · seed {0}", 10),
    (r"^t1s(\d)_ft_canon_bfi_n_", "yolo26n · dense -> canonical fine-tune · seed {0}", 11),
    (r"^t1s(\d)_ftlow_canon_bfi_n_", "yolo26n · dense -> canonical fine-tune (low lr) · seed {0}", 11),
]


def _friendly(stem):
    for rx, tpl, rank in _NAMES:
        m = re.match(rx, stem)
        if m:
            return tpl.format(*m.groups()), rank
    return stem, 99


def load_registry():
    """stem -> {label, rank, sizes: {imgsz: {onnx, params_M, gflops, onnx_MB, conf, test_f2}}}, paper students first."""
    by_key = {}
    for r in _read_tsv(MANIFEST_TSV):                       # later rows win (re-exports)
        sz = _f(r.get("imgsz"))
        p = r.get("onnx_path", "")
        if sz is None or not p:
            continue
        p_local = os.path.join(EXPORTS_DIR, os.path.basename(p))   # EXPORTS_DIR wins; the manifest's own path is a fallback ONLY in the
        p = p_local if os.path.isfile(p_local) else (p if (os.path.isfile(p) and "RIPAPP_EXPORTS_DIR" not in os.environ) else None)   # default layout
        if p is None:
            continue
        by_key[(r["stem"], int(sz))] = dict(onnx=p, params_M=_f(r.get("params_M")), gflops=_f(r.get("gflops")), onnx_MB=_f(r.get("onnx_MB")))
    if not by_key and os.path.isdir(EXPORTS_DIR):           # no manifest at all: scan the exports directory
        for fn in sorted(os.listdir(EXPORTS_DIR)):
            m = re.match(r"^(.+)_(640|512|384)\.onnx$", fn)
            if m and "gate" not in fn and "fp16" not in fn:
                by_key[(m.group(1), int(m.group(2)))] = dict(onnx=os.path.join(EXPORTS_DIR, fn), params_M=None, gflops=None, onnx_MB=None)
    val_conf, test_f2 = {}, {}
    for r in _read_tsv(EVAL_TSV):
        if r.get("split") == "val":
            val_conf[r["model"]] = _f(r.get("selected_conf"))
        elif r.get("split") == "test":
            test_f2[r["model"]] = _f(r.get("F2_50"))
    reg = {}
    for (stem, sz), info in by_key.items():
        key = stem if sz == 640 else f"{stem}_img{sz}"      # eval rows: 640 = bare stem, other sizes = _img<sz>
        info["conf"], info["test_f2"] = val_conf.get(key), test_f2.get(key)
        info["family"], info["gt_view"] = "yolo", "bfi"      # students are scored on the bbox_from_instance GT view
        label, rank = _friendly(stem)
        reg.setdefault(stem, dict(stem=stem, label=label, rank=rank, sizes={}))["sizes"][sz] = info
    for stem, h in HEAVY_MODELS.items():
        if os.path.isfile(h["onnx"]):
            reg[stem] = dict(stem=stem, label=h["label"], rank=3.5, sizes={h["imgsz"]: dict(
                onnx=h["onnx"], params_M=h["params_M"], gflops=h["gflops"], onnx_MB=round(os.path.getsize(h["onnx"]) / 1e6, 1),
                conf=h["conf"], test_f2=h["test_f2"], family=h["family"], gt_view=h["gt_view"], rip_index=h.get("rip_index", 0))})
    return OrderedDict(sorted(reg.items(), key=lambda kv: (kv[1]["rank"], kv[1]["label"])))


# ---------- inference backends ----------
class OrtRunner:
    def __init__(self, path, providers, threads):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, so, providers=providers)
        self.name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        prov = self.sess.get_providers()[0]
        self.short = {"CPUExecutionProvider": "ORT CPU", "CUDAExecutionProvider": "ORT CUDA"}.get(prov, prov)
        self.desc = f"onnxruntime {ort.__version__} / {prov} / {threads} threads"

    def infer(self, x):
        return self.sess.run(None, {self.name: x})


class EngineRunner:
    """TensorRT engine produced by jetson_measure.build_engine (trtexec); needs tensorrt + pycuda (JetPack)."""
    def __init__(self, engine):
        from jetson_measure import TrtRunner
        self.r = TrtRunner(engine)
        self.out_names = self.r.out_names
        prec = os.path.splitext(engine)[0].rsplit("_", 1)[-1].upper()          # jetson_measure naming: <stem>_<precision>.engine
        self.short, self.desc = f"TensorRT {prec}", f"TensorRT {prec} engine {os.path.basename(engine)}"

    def infer(self, x):
        return self.r.infer(x)


def engine_path(onnx_path):
    return os.path.join(ENGINE_DIR, f"{os.path.splitext(os.path.basename(onnx_path))[0]}_{ENGINE_PRECISION}.engine")


def make_runner(onnx_path, backend, log=print):
    import onnxruntime as ort
    if backend in ("auto", "trt-engine"):
        eng = engine_path(onnx_path)
        if os.path.isfile(eng):
            try:
                return EngineRunner(eng)
            except Exception as e:                            # noqa: BLE001
                log(f"TensorRT engine {eng} unusable ({e}); falling back to onnxruntime")
        elif backend == "trt-engine":
            raise FileNotFoundError(f"no TensorRT engine {eng} — build it with jetson_measure.build_engine / trtexec")
    if backend == "cuda":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return OrtRunner(onnx_path, ["CUDAExecutionProvider", "CPUExecutionProvider"], THREADS)
        log("CUDAExecutionProvider not available in this onnxruntime build; using CPU")
    return OrtRunner(onnx_path, ["CPUExecutionProvider"], THREADS)


# ---------- gate (copied from hil_demo.gate_prob so a non-ORT runner can be used) ----------
def gate_input(frame):
    return classify_preprocess(frame, GATE_IMGSZ)[0]   # the board runtime's gate path: short side to 224 px, centre crop


def gate_prob(out):
    o = np.asarray(out[0]).reshape(-1).astype(np.float64)
    if not (0.99 <= o.sum() <= 1.01 and o.min() >= 0):        # ultralytics classify export already emits softmax probs
        o = np.exp(o - o.max())
        o /= o.sum()
    return float(o[1])                                          # class order {0: no_rip, 1: rip}


# ---------- board power (Jetson INA3221 rails via tegrastats; "n/a" elsewhere) ----------
class _SimTegrastats:
    """TEST HOOK ONLY (RIPAPP_POWER_SIM=1, hosts without tegrastats): fake rails = 4 W + 3 W x this process's CPU busy fraction."""
    def __init__(self):
        self.samples, self._stop = [], threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cp0, wt0 = time.process_time(), time.time()
        while not self._stop.is_set():
            time.sleep(0.1)
            cp, wt = time.process_time(), time.time()
            busy = min(1.0, (cp - cp0) / max(1e-6, wt - wt0) / THREADS)
            cp0, wt0 = cp, wt
            mw = int(1000 * (4.0 + 3.0 * busy))
            self.samples.append((wt, {"VDD_IN": mw, "VDD_CPU_GPU_CV": int(mw * 0.6), "VDD_SOC": int(mw * 0.3)}))

    def stop(self):
        self._stop.set()
        return list(self.samples)


class PowerMonitor:
    """calibrate(): an IDLE window (models loaded, nothing running) and a continuous-inference BURST window -> idle_W /
    active_W — the measurement protocol's two states in miniature; energy per decision = (active_W - idle_W) x t_decision
    (jetson_measure.py owns the real 200/1000-frame protocol; these are the demo's live readouts)."""
    def __init__(self):
        self.sim = bool(os.environ.get("RIPAPP_POWER_SIM"))
        self.ok = self.sim or shutil.which("tegrastats") is not None
        self.ts, self.idle_w, self.active_w = None, None, None
        self.idle_win, self.burst_win = None, None               # (t0, t1[, n_inferences]) — provenance for the energy numbers

    def start(self, log=print):
        if not self.ok:
            return
        try:
            if self.sim:
                self.ts = _SimTegrastats()
            else:
                from jetson_measure import Tegrastats
                self.ts = Tegrastats()
            self.ts.start()
        except Exception as e:                                  # noqa: BLE001
            log(f"tegrastats unavailable ({e}); power = n/a")
            self.ok, self.ts = False, None

    def mean_w(self, t0, t1=None, rail="VDD_IN"):
        if not self.ts:
            return None
        t1 = t1 or time.time()
        v = [smp[rail] for t, smp in list(self.ts.samples) if t0 <= t <= t1 and rail in smp]
        return float(np.mean(v)) / 1000.0 if v else None

    def calibrate(self, burst_fn, log=print):
        if not self.ts:
            return
        t0 = time.time()
        time.sleep(POWER_IDLE_SECONDS)
        self.idle_w = self.mean_w(t0)
        self.idle_win = (t0, time.time())
        t0, n = time.time(), 0
        while time.time() - t0 < POWER_BURST_SECONDS:
            burst_fn()
            n += 1
        self.active_w = self.mean_w(t0)
        self.burst_win = (t0, time.time(), n)
        if self.idle_w is not None and self.active_w is not None:
            log(f"power: idle {self.idle_w:.2f} W ({POWER_IDLE_SECONDS:g} s), active {self.active_w:.2f} W ({n} inferences in {POWER_BURST_SECONDS:g} s)"
                + (" [SIMULATED rails]" if self.sim else ""))

    def now_w(self):
        if self.ts and self.ts.samples:
            rails = self.ts.samples[-1][1]
            if "VDD_IN" in rails:
                return rails["VDD_IN"] / 1000.0
        return None

    def mj_per_decision(self, decision_ms):
        if self.idle_w is None or self.active_w is None:
            return None
        return max(0.0, self.active_w - self.idle_w) * decision_ms      # W x ms = mJ

    def stop(self, run_t0=None):
        if not self.ts:
            return {}
        try:
            from jetson_measure import TEGRASTATS_INTERVAL_MS as interval_ms
        except Exception:                                       # noqa: BLE001
            interval_ms = None
        out = {"source": "SIMULATED (RIPAPP_POWER_SIM)" if self.sim else "tegrastats VDD_IN (INA3221)",
               "sample_interval_ms": 100 if self.sim else interval_ms, "rail": "VDD_IN",
               "idle_W": self.idle_w, "active_W": self.active_w,
               "idle_window": dict(start=self.idle_win[0], end=self.idle_win[1], seconds=round(self.idle_win[1] - self.idle_win[0], 2),
                                   state="models loaded, no inference") if self.idle_win else None,
               "active_window": dict(start=self.burst_win[0], end=self.burst_win[1], seconds=round(self.burst_win[1] - self.burst_win[0], 2),
                                     inferences=self.burst_win[2], state="continuous gate+detector inference") if self.burst_win else None,
               "mJ_per_decision_formula": "(active_W - idle_W) x (gate_ms + detector_ms)"}
        if run_t0 is not None:
            out["mean_W_run"] = self.mean_w(run_t0)
            out["mean_W_run_VDD_CPU_GPU_CV"] = self.mean_w(run_t0, rail="VDD_CPU_GPU_CV")
        samples = self.ts.stop()
        self.ts = None
        out["n_samples"] = len(samples)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


# ---------- video sources ----------
class Source:
    """file: sequential frames, video time = n/fps.  live (webcam / stream): a reader thread keeps only the newest
    frame so a slow board never accumulates latency (dropped frames are counted in `grabbed`)."""
    def __init__(self, kind, value):
        self.kind = kind
        if kind == "webcam":
            idx = int(value)
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(idx)
            self.name, self.live, self.value = f"webcam{idx}", True, idx
        elif kind == "url":
            if value.lower().startswith("rtsp") and RTSP_TRANSPORT:     # OpenCV reads this env var at open time
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{RTSP_TRANSPORT}"
            else:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            self.cap = cv2.VideoCapture(value, cv2.CAP_FFMPEG)
            self.name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.split("://")[-1])[:40].strip("_")
            self.live, self.value = not value.lower().split("?")[0].endswith(VIDEO_EXT), value
        else:
            self.cap = cv2.VideoCapture(value)
            self.name, self.live, self.value = os.path.splitext(os.path.basename(value))[0], False, value
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {kind} source: {value}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (1.0 <= self.fps <= 240.0):
            self.fps = 30.0
        self.W, self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = 0 if self.live else max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.grabbed, self._served, self._latest = 0, -1, None
        self._lock, self._stop = threading.Lock(), threading.Event()
        if self.live:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:                                   # noqa: BLE001
                pass
            self._th = threading.Thread(target=self._reader, daemon=True)
            self._th.start()

    def _reader(self):
        while not self._stop.is_set():
            ok, fr = self.cap.read()
            if not ok:                                          # stream ended / camera unplugged
                self._stop.set()
                break
            with self._lock:
                self._latest, self.grabbed = fr, self.grabbed + 1

    def read(self, timeout=5.0):
        """-> (frame, index) or (None, None) at the end of the source."""
        if not self.live:
            ok, fr = self.cap.read()
            if not ok:
                return None, None
            self.grabbed += 1
            return fr, self.grabbed - 1
        t0 = time.time()
        while True:
            with self._lock:
                fr, idx = self._latest, self.grabbed - 1
            if fr is not None and idx > self._served:
                self._served = idx
                return fr, idx
            if self._stop.is_set() or time.time() - t0 > timeout:
                return None, None
            time.sleep(0.002)

    def release(self):
        self._stop.set()
        if self.live:
            self._th.join(timeout=2.0)
        self.cap.release()


# ---------- overlay ----------
def _text(vis, s, org, fs, color, th):
    cv2.putText(vis, s, org, FONT, fs, color, th, cv2.LINE_AA)


def draw_overlay(vis, st, boxes, scores):
    H, W = vis.shape[:2]
    fs = max(0.5, HUD_SCALE * H / 1080.0)
    th = max(1, int(round(fs * 1.6)))
    lh = int(38 * fs)
    bt = max(2, int(3 * fs))
    lfs = max(0.5, fs * 0.8)                                  # box labels
    order = np.argsort(np.asarray(scores)) if len(scores) else []     # strongest box drawn last (its label stays on top)
    for b, sc in ((boxes[i], scores[i]) for i in order):
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
        cv2.rectangle(vis, (x1, y1), (x2, y2), RED, bt)
        lab = f"rip {sc:.2f}"
        (tw, tH), _ = cv2.getTextSize(lab, FONT, lfs, th)
        y = y1 - 6 if y1 - 6 - tH > 0 else y2 + tH + 6
        cv2.rectangle(vis, (x1, y - tH - 6), (x1 + tw + 10, y + 6), RED, -1)
        _text(vis, lab, (x1 + 5, y), lfs, WHITE, th)
    if st["gate_on"]:
        g_col = GREEN if st["gate_open"] else GREY
        g_line = f"GATE   P(rip) {st['gate_p']:.2f}  {'OPEN' if st['gate_open'] else 'closed'}  {st['gate_ms']:.1f} ms"
    else:
        g_col, g_line = AMBER, "GATE   off (detector at every decision)"
    d_col = RED if st["n_det"] else (WHITE if st["det_ran"] else GREY)
    d_line = (f"DET    conf {st['conf']:.2f}  {st['det_runs']}/{st['ticks']} ({st['duty']:.0%})  "
              + (f"{st['det_ms']:.1f} ms  {st['n_det']} rip" + ("s" if st["n_det"] != 1 else "") if st["det_ran"] else "skipped (gate closed)"))
    if st["power_w"] is not None:
        p_line = f"POWER  {st['power_w']:.1f} W"
        if st.get("idle_w") is not None and st.get("active_w") is not None:
            p_line += f"  (idle {st['idle_w']:.1f} / active {st['active_w']:.1f})"
        if st.get("mj") is not None:
            p_line += f"  {st['mj']:.0f} mJ/decision"
    else:
        p_line = "POWER  n/a (no tegrastats on this host)"
    lines = [
        (st["title"], WHITE),
        (f"t {st['t']:.1f} s   {st['proc_fps']:.1f} fps   {st['duty_fps']:g} decision/s", WHITE),
        (g_line, g_col), (d_line, d_col), (p_line, WHITE),
    ]
    tw = max(cv2.getTextSize(s, FONT, fs, th)[0][0] for s, _ in lines)
    pad = int(12 * fs)
    if tw + 2 * pad > W:                                    # narrow (portrait / low-res) frames: shrink the HUD to fit
        fs = max(0.35, fs * (W - 2 * pad) / tw)
        th, lh, pad = max(1, int(round(fs * 1.6))), int(38 * fs), int(12 * fs)
        tw = max(cv2.getTextSize(s, FONT, fs, th)[0][0] for s, _ in lines)
    ph = pad * 2 + lh * len(lines)
    pw = min(W, tw + 2 * pad)
    roi = vis[0:ph, 0:pw]
    cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0, roi)
    y = pad + int(lh * 0.75)
    for s, col in lines:
        _text(vis, s, (pad, y), fs, col, th)
        y += lh
    if st["n_det"]:
        lab = "RIP CURRENT"
        (bw, bh), _ = cv2.getTextSize(lab, FONT, fs * 1.2, th + 1)
        cv2.rectangle(vis, (W - bw - 3 * pad, pad), (W - pad, pad * 2 + bh + pad), RED, -1)
        _text(vis, lab, (W - bw - 2 * pad, pad * 2 + bh), fs * 1.2, WHITE, th + 1)
    return vis


def _hud_name(label):
    """'yolo26n · dense supervision (paper student) · seed 1' -> 'yolo26n dense supervision' (ASCII, short, for the HUD title)."""
    return _ascii(re.sub(r"\s*\([^)]*\)", "", re.sub(r"\s*·\s*seed \d+", "", label))).replace(" - ", " ").strip()


def _ascii(s):
    return s.replace(" · ", " - ").replace("·", "-").replace("→", "->").replace("—", "-")   # cv2.putText draws only ASCII


def _resize_w(img, max_w):
    if max_w and img.shape[1] > max_w:
        h = int(round(img.shape[0] * max_w / img.shape[1]))
        return cv2.resize(img, (max_w, h), interpolation=cv2.INTER_AREA)
    return img


def _git_hash():
    try:
        return subprocess.run(["git", "-C", EDGE_DIR, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:                                           # noqa: BLE001
        return None


def transcode_h264(src, dst, log=print):
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    cmd = [ff, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"ffmpeg transcode failed: {r.stderr.strip()[:200]}")
        return False
    return True


# ---------- the pipeline (one generator, used by the UI and the headless mode) ----------
def run_pipeline(cfg, registry=None, stop_event=None, log=print, want_display=True):
    """cfg: source_kind (file|url|webcam), source, model, imgsz, backend, duty_fps, gate, gate_thr, conf (None = val-frozen),
    record, pace_realtime, max_seconds (0 = whole), out_name.  Yields dicts: {"frame": RGB display image or None,
    "stats": {...}} at DISPLAY_FPS, then a final {"done": True, "summary": {...}, "paths": {...}}."""
    stop_event = stop_event or threading.Event()
    registry = registry or load_registry()
    cfg = {**HEADLESS_DEFAULTS, **cfg}
    stem, imgsz = cfg["model"], int(cfg["imgsz"])
    if stem not in registry or imgsz not in registry[stem]["sizes"]:
        raise ValueError(f"model {stem} @ {imgsz} not in the export manifest ({MANIFEST_TSV})")
    minfo = registry[stem]["sizes"][imgsz]
    label = registry[stem]["label"]
    conf_frozen = minfo["conf"]
    conf = float(cfg["conf"]) if cfg.get("conf") is not None else (conf_frozen if conf_frozen is not None else 0.25)
    conf_is_frozen = cfg.get("conf") is None and conf_frozen is not None
    gate_on, gate_thr, duty_fps = bool(cfg["gate"]), float(cfg["gate_thr"]), float(cfg["duty_fps"])
    max_s = float(cfg.get("max_seconds") or 0)

    family = minfo.get("family", "yolo")
    log(f"model {label} [{stem}] @ {imgsz}px  conf {conf:.2f}{' (val-frozen)' if conf_is_frozen else ' (manual)'}  gate {'on thr %.2f' % gate_thr if gate_on else 'off'}  duty {duty_fps:g}/s"
        + (f"  [{family}, benchmark GT view {minfo.get('gt_view')}]"))
    det = make_runner(minfo["onnx"], cfg["backend"], log)
    if family == "rfdetr":
        import rfdetr_onnx_bench as RB                      # the ICRA session's validated pre/post-processing (imported, never modified)
        assert RB.RES == imgsz, f"rfdetr_onnx_bench.RES {RB.RES} != {imgsz}"
        rip_index = minfo.get("rip_index", 0)

        def detect(frame):                                   # DETR outputs: top-K over queries, no NMS
            x, w, h = RB.preprocess(frame)
            res = dict(zip(det.out_names, det.infer(x)))
            dd = RB.decode(res["dets"], res["labels"], w, h, rip_index, 1)
            b = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]] for d in dd], np.float64).reshape(-1, 4)
            sc = np.array([d["score"] for d in dd], np.float64)
            keep = sc >= conf
            return b[keep], sc[keep], int(keep.sum())
    else:
        def detect(frame):                                   # ultralytics end-to-end export: board_bench decode + NMS 0.7 (benchmark convention)
            x, r, top, left = letterbox(frame, imgsz)
            b, sc = decode(det.infer(x), r, top, left, conf)
            n_raw = len(b)
            if NMS_IOU and len(b) > 1:
                k = nms(b, sc, NMS_IOU)
                b, sc = b[k], sc[k]
            return b, sc, n_raw
    gate = make_runner(GATE_ONNX, cfg["backend"], log) if gate_on else None
    for _ in range(3):                                          # warm-up
        det.infer(np.zeros((1, 3, imgsz, imgsz), np.float32))
        if gate:
            gate.infer(np.zeros((1, 3, GATE_IMGSZ, GATE_IMGSZ), np.float32))
    log(f"detector backend: {det.desc}" + (f" | gate: {gate.desc}" if gate else ""))

    src = Source(cfg["source_kind"], cfg["source"])
    log(f"source {src.name}: {src.W}x{src.H} @ {src.fps:.2f} fps, {'live' if src.live else f'{src.n_frames} frames'}")
    period = 1.0 / duty_fps
    step = max(1, int(round(src.fps / duty_fps)))              # file replay: one decision every `step` frames (as hil_demo.py)
    rec_fps = LIVE_LOOP_FPS if src.live else src.fps

    out_dir = tsv = writer = None
    paths = {}
    if cfg["record"]:
        name = cfg.get("out_name") or f"{datetime.now():%Y%m%d_%H%M%S}_{src.name}_{stem}_{imgsz}"
        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        paths = dict(dir=out_dir, mp4=os.path.join(out_dir, "overlay.mp4"), tsv=os.path.join(out_dir, "frames.tsv"), json=os.path.join(out_dir, "summary.json"))
        tsv = open(paths["tsv"], "w")
        tsv.write("frame\ttime_s\twall_time\ttick\tgate_p\tgate_open\tdet_ran\tn_det\tn_det_raw\tgate_ms\tdet_ms\tpower_w\tmj_decision\tboxes_xyxy_score\n")
        log(f"recording -> {out_dir}")
    power = PowerMonitor()
    power.start(log)

    def _burst():
        det.infer(np.zeros((1, 3, imgsz, imgsz), np.float32))
        if gate:
            gate.infer(np.zeros((1, 3, GATE_IMGSZ, GATE_IMGSZ), np.float32))
    power.calibrate(_burst, log)
    rip_window = cfg.get("rip_window", RIP_WINDOWS.get(src.name))
    rip_window = tuple(rip_window) if rip_window else None

    st = dict(title=f"{_hud_name(label)} | {imgsz} px | {det.short}", t=0.0, frame=0, proc_fps=0.0, duty_fps=duty_fps,
              gate_on=gate_on, gate_p=0.0, gate_open=not gate_on, gate_ms=0.0, det_ran=False, det_ms=0.0, n_det=0, n_raw=0,
              det_runs=0, ticks=0, duty=0.0, power_w=None, idle_w=power.idle_w, active_w=power.active_w, mj=None,
              source=src.name, backend=det.desc, model=label, stem=stem, imgsz=imgsz, conf=conf, recording=out_dir or "",
              grabbed=0, rip_ticks=0, first_rip_t=None, false_alarms=0 if rip_window else None, rip_window=rip_window)
    boxes, scores = np.zeros((0, 4)), np.zeros(0)
    gate_ms_all, det_ms_all, mj_all, loop_times = [], [], [], deque(maxlen=60)
    n_proc, wall0, next_tick, last_disp, last_log = 0, time.time(), 0.0, 0.0, 0.0
    t_video = 0.0
    try:
        while not stop_event.is_set():
            frame, idx = src.read()
            if frame is None:
                break
            now = time.time()
            t_video = idx / src.fps if not src.live else now - wall0
            if max_s and t_video > max_s:
                break
            is_tick = (idx % step == 0) if not src.live else (now >= next_tick)
            if is_tick:
                if src.live:
                    next_tick = now + period
                st["ticks"] += 1
                t0 = time.perf_counter()
                if gate:
                    p = gate_prob(gate.infer(gate_input(frame)))
                    st["gate_p"], st["gate_open"] = p, p >= gate_thr
                    st["gate_ms"] = (time.perf_counter() - t0) * 1e3
                    gate_ms_all.append(st["gate_ms"])
                st["det_ran"] = st["gate_open"]
                if st["det_ran"]:
                    t1 = time.perf_counter()
                    boxes, scores, st["n_raw"] = detect(frame)
                    st["det_ms"] = (time.perf_counter() - t1) * 1e3
                    det_ms_all.append(st["det_ms"])
                    st["det_runs"] += 1
                else:
                    boxes, scores = np.zeros((0, 4)), np.zeros(0)
                    st["n_raw"] = 0
                st["n_det"] = len(boxes)
                st["duty"] = st["det_runs"] / st["ticks"]
                decision_ms = (st["gate_ms"] if gate else 0.0) + (st["det_ms"] if st["det_ran"] else 0.0)
                st["mj"] = power.mj_per_decision(decision_ms)
                if st["mj"] is not None:
                    mj_all.append(st["mj"])
                if len(boxes):
                    st["rip_ticks"] += 1
                    if st["first_rip_t"] is None:
                        st["first_rip_t"] = round(t_video, 2)
                    if rip_window and not (rip_window[0] <= t_video <= (rip_window[1] if rip_window[1] is not None else float("inf"))):
                        st["false_alarms"] += 1
            st["t"], st["frame"], st["grabbed"], st["power_w"] = t_video, idx, src.grabbed, power.now_w()
            loop_times.append(now)
            if len(loop_times) > 1:
                st["proc_fps"] = (len(loop_times) - 1) / max(1e-6, loop_times[-1] - loop_times[0])
            vis = draw_overlay(frame.copy(), st, boxes, scores)
            if tsv:
                bx = json.dumps([[round(float(v), 1) for v in b] + [round(float(s), 3)] for b, s in zip(boxes, scores)])
                pw_s = "" if st["power_w"] is None else f"{st['power_w']:.2f}"
                mj_s = "" if st["mj"] is None else f"{st['mj']:.1f}"
                if is_tick:
                    tsv.write(f"{idx}\t{t_video:.3f}\t{now:.3f}\t1\t{st['gate_p']:.4f}\t{int(st['gate_open'])}\t{int(st['det_ran'])}\t{len(boxes)}\t{st['n_raw']}\t"
                              f"{st['gate_ms']:.2f}\t{st['det_ms'] if st['det_ran'] else 0:.2f}\t{pw_s}\t{mj_s}\t{bx}\n")
                else:
                    tsv.write(f"{idx}\t{t_video:.3f}\t{now:.3f}\t0\t\t\t\t{len(boxes)}\t\t\t\t{pw_s}\t\t{bx}\n")
                if writer is None:
                    rec = _resize_w(vis, RECORD_MAX_WIDTH)
                    paths["mp4_raw"] = os.path.join(out_dir, "overlay_raw.mp4")
                    writer = cv2.VideoWriter(paths["mp4_raw"], cv2.VideoWriter_fourcc(*"mp4v"), rec_fps, (rec.shape[1], rec.shape[0]))
                    writer_size = (rec.shape[1], rec.shape[0])
                rec = _resize_w(vis, RECORD_MAX_WIDTH)
                if (rec.shape[1], rec.shape[0]) != writer_size:
                    rec = cv2.resize(rec, writer_size)
                writer.write(rec)
            n_proc += 1
            if want_display and now - last_disp >= 1.0 / DISPLAY_FPS:
                last_disp = now
                disp = _resize_w(vis, DISPLAY_MAX_WIDTH)
                yield {"frame": np.ascontiguousarray(disp[:, :, ::-1]), "stats": dict(st)}
            if now - last_log >= 5.0:
                last_log = now
                log(f"t={t_video:6.1f}s frame {idx} ticks {st['ticks']} det-runs {st['det_runs']} ({st['duty']:.0%}) rip-ticks {st['rip_ticks']} proc {st['proc_fps']:.1f} fps")
            if src.live:                                        # pace the loop between ticks; ticks themselves are wall-clock scheduled
                time.sleep(max(0.0, 1.0 / LIVE_LOOP_FPS - (time.time() - now)))
            elif cfg["pace_realtime"]:
                time.sleep(max(0.0, wall0 + (idx + 1) / src.fps - time.time()))
    finally:
        wall = time.time() - wall0
        src.release()
        if writer is not None:
            writer.release()
        if tsv:
            tsv.close()
        pw = power.stop(wall0)
        if writer is not None and TRANSCODE_H264 and transcode_h264(paths["mp4_raw"], paths["mp4"], log):
            os.remove(paths["mp4_raw"])
        elif writer is not None:
            os.replace(paths["mp4_raw"], paths["mp4"])          # mp4v stays (playable in VLC, not in browsers)
        paths.pop("mp4_raw", None)
        summary = dict(
            app_version=APP_VERSION, git=_git_hash(), started=datetime.fromtimestamp(wall0).isoformat(timespec="seconds"),
            ended=datetime.now().isoformat(timespec="seconds"), wall_seconds=round(wall, 2), stopped_early=stop_event.is_set(),
            host=dict(node=platform.node(), platform=platform.platform(), python=platform.python_version()),
            source=dict(kind=src.kind, value=str(src.value), name=src.name, fps=round(src.fps, 3), width=src.W, height=src.H,
                        n_frames=src.n_frames, live=src.live),
            model=dict(stem=stem, label=label, family=family, imgsz=imgsz, onnx=minfo["onnx"], params_M=minfo["params_M"], gflops=minfo["gflops"],
                       conf=conf, conf_is_val_frozen=conf_is_frozen, val_frozen_conf=conf_frozen, test_F2_50_benchmark=minfo["test_f2"],
                       benchmark_gt_view=minfo.get("gt_view"), nms_iou=NMS_IOU if family == "yolo" else None),
            gate=dict(enabled=gate_on, onnx=GATE_ONNX if gate_on else None, thr=gate_thr, imgsz=GATE_IMGSZ), nms_iou=NMS_IOU,
            backend=dict(detector=det.desc, gate=gate.desc if gate else None, threads=THREADS),
            duty_fps=duty_fps, frames_processed=n_proc, frames_grabbed=src.grabbed, video_seconds=round(t_video, 2),
            realtime_factor=round(t_video / wall, 3) if wall > 0 else None, ticks=st["ticks"], detector_runs=st["det_runs"],
            detector_duty=round(st["duty"], 4), rip_ticks=st["rip_ticks"], first_rip_time_s=st["first_rip_t"],
            rip_window_s=list(rip_window) if rip_window else None, false_alarm_ticks=st["false_alarms"],
            gate_ms=dict(mean=round(float(np.mean(gate_ms_all)), 2), p90=round(float(np.percentile(gate_ms_all, 90)), 2)) if gate_ms_all else None,
            det_ms=dict(mean=round(float(np.mean(det_ms_all)), 2), p90=round(float(np.percentile(det_ms_all, 90)), 2)) if det_ms_all else None,
            proc_fps_mean=round(n_proc / wall, 2) if wall > 0 else None,
            power=pw or None,
            mJ_per_decision=dict(mean=round(float(np.mean(mj_all)), 1), p90=round(float(np.percentile(mj_all, 90)), 1)) if mj_all else None,
            energy_Wh_run=round(pw["mean_W_run"] * wall / 3600.0, 4) if pw and pw.get("mean_W_run") else None,
            outputs=paths or None,
        )
        if paths:
            with open(paths["json"], "w") as f:
                json.dump(summary, f, indent=1)
        log(f"done: {n_proc} frames, {st['ticks']} decisions, detector ran {st['det_runs']} ({st['duty']:.0%}), rip on {st['rip_ticks']}"
            + (f", false alarms {st['false_alarms']}" if st["false_alarms"] is not None else "")
            + (f", {np.mean(mj_all):.0f} mJ/decision" if mj_all else "") + f", {wall:.1f} s wall" + (f" -> {out_dir}" if out_dir else ""))
    yield {"done": True, "summary": summary, "paths": paths, "stats": dict(st)}


# ---------- headless mode ----------
def run_headless(jobs):
    reg = load_registry()
    results = []
    for i, job in enumerate(jobs):
        print(f"=== job {i + 1}/{len(jobs)}: {job}", flush=True)
        last = None
        for ev in run_pipeline(job, reg, log=lambda s: print(s, flush=True), want_display=False):
            last = ev
        results.append(last["summary"] if last else None)
    print("RIPAPP_HEADLESS_DONE", json.dumps([r["outputs"] for r in results if r], indent=1), flush=True)
    return results


# ---------- Gradio UI (a thin layer over run_pipeline) ----------
def _stats_html(st, extra=""):
    if not st:
        return "<p style='color:#888'>idle</p>"
    rows = [
        ("Model", f"{st['model']} @ {st['imgsz']}px · conf {st['conf']:.2f}"), ("Backend", st["backend"]), ("Source", st["source"]),
        ("Video time / frame", f"{st['t']:.1f} s / {st['frame']}" + (f" (grabbed {st['grabbed']})" if st["grabbed"] > st["frame"] + 1 else "")),
        ("Processing", f"{st['proc_fps']:.1f} fps"),
        ("Gate", (f"P(rip) {st['gate_p']:.2f} — <b style='color:{'#2a2' if st['gate_open'] else '#888'}'>{'OPEN' if st['gate_open'] else 'closed'}</b> · {st['gate_ms']:.1f} ms") if st["gate_on"] else "off"),
        ("Detector", f"ran {st['det_runs']}/{st['ticks']} ticks (duty {st['duty']:.0%}) · {st['det_ms']:.1f} ms · "
                     + (f"<b style='color:#d22'>{st['n_det']} rip(s)</b>" if st["n_det"] else "no rip")),
        ("Rip decisions", f"{st['rip_ticks']}" + (f" (first at {st['first_rip_t']} s)" if st["first_rip_t"] is not None else "")
                          + (f" · false alarms {st['false_alarms']} (rip window {st['rip_window'][0]:g}–{st['rip_window'][1] if st['rip_window'][1] is not None else 'end'} s)" if st.get("rip_window") else "")),
        ("Power", (f"{st['power_w']:.2f} W (VDD_IN)" + (f" · idle {st['idle_w']:.2f} / active {st['active_w']:.2f} W" if st.get("idle_w") is not None and st.get("active_w") is not None else "")
                   + (f" · {st['mj']:.0f} mJ per decision" if st.get("mj") is not None else "")) if st["power_w"] is not None else "n/a (no tegrastats)"),
        ("Recording", st["recording"] or "off"),
    ]
    tr = "".join(f"<tr><td style='padding:2px 10px 2px 0;color:#888;white-space:nowrap'>{k}</td><td style='padding:2px 0'>{v}</td></tr>" for k, v in rows)
    return f"{extra}<table style='font-size:13px;line-height:1.35'>{tr}</table>"     # summary line first: it must stay above the fold


def build_ui(reg):
    import gradio as gr
    kinds = ["Sample clip", "Video file", "Stream URL", "Webcam"]
    samples = [(k, v) for k, v in SAMPLE_CLIPS.items() if os.path.isfile(v)]
    choices = [(f"{info['label']}   [{stem}]", stem) for stem, info in reg.items()]
    stem0 = DEFAULT_MODEL if DEFAULT_MODEL in reg else next(iter(reg))
    sizes0 = sorted(reg[stem0]["sizes"], reverse=True)
    sz0 = DEFAULT_IMGSZ if DEFAULT_IMGSZ in sizes0 else sizes0[0]

    def conf_of(stem, sz):
        info = reg.get(stem, {}).get("sizes", {}).get(int(sz)) if stem else None
        if not info:
            return 0.25, "no export for this model/size"
        c, f2 = info["conf"], info["test_f2"]
        if c is None:
            return 0.25, "no val-frozen conf for this model/size (default 0.25)"
        return c, (f"val-frozen conf **{c:.2f}**" + (f" · benchmark test F2@50 {f2:.3f} ({info.get('gt_view', '?')} GT view)" if f2 is not None else "")
                   + (f" · {info['gflops']:.1f} GFLOPs" if info.get("gflops") else (f" · {info['params_M']:.0f} M params" if info.get("params_M") else "")))

    def on_model(stem, sz):
        sizes = sorted(reg[stem]["sizes"], reverse=True)
        sz = int(sz) if int(sz) in sizes else sizes[0]
        c, txt = conf_of(stem, sz)
        return gr.update(choices=sizes, value=sz), gr.update(value=c), txt

    def on_size(stem, sz):
        c, txt = conf_of(stem, sz)
        return gr.update(value=c), txt

    def on_kind(k):
        return [gr.update(visible=(k == x)) for x in kinds]

    def ui_stop():
        STOP_EVENT.set()
        return "stop requested"

    def ui_run(kind, sample, upload, url, cam, stem, sz, backend, duty, gate_on, gate_thr, conf, record, pace, max_s):
        if not RUN_LOCK.acquire(blocking=False):
            raise gr.Error("a run is already active — press Stop first")
        lines = []

        def log(s):
            lines.append(f"{datetime.now():%H:%M:%S}  {s}")
            print(s, flush=True)
        try:
            STOP_EVENT.clear()
            if kind == "Sample clip":
                sk, sv = "file", sample
            elif kind == "Video file":
                sk, sv = "file", upload
            elif kind == "Stream URL":
                sk, sv = "url", (url or "").strip()
            else:
                sk, sv = "webcam", int(cam or 0)
            if sv in (None, ""):
                raise gr.Error("pick a source first")
            sizes = reg[stem]["sizes"]
            frozen = sizes[int(sz)]["conf"] if int(sz) in sizes else None
            cfg = dict(source_kind=sk, source=sv, model=stem, imgsz=int(sz), backend=backend, duty_fps=float(duty), gate=bool(gate_on),
                       gate_thr=float(gate_thr), conf=None if (frozen is not None and abs(float(conf) - frozen) < 1e-6) else float(conf),
                       record=bool(record), pace_realtime=bool(pace), max_seconds=float(max_s or 0))
            keep = gr.update()
            yield None, _stats_html(None), keep, keep, keep, "starting…"
            try:
                for ev in run_pipeline(cfg, reg, STOP_EVENT, log, want_display=True):
                    if ev.get("done"):
                        s, p = ev["summary"], ev["paths"]
                        files = [p[k] for k in ("mp4", "tsv", "json") if k in p and os.path.isfile(p[k])]
                        extra = (f"<p style='margin:0 0 6px 0;font-size:14px'><b>done</b> — {s['frames_processed']} frames, {s['ticks']} decisions, detector duty "
                                 f"{s['detector_duty']:.0%}, rip on {s['rip_ticks']} decisions"
                                 + (f", false alarms {s['false_alarm_ticks']}" if s["false_alarm_ticks"] is not None else "")
                                 + (f", {s['mJ_per_decision']['mean']:.0f} mJ/decision, {s['power']['mean_W_run']:.2f} W avg" if s.get("mJ_per_decision") and s["power"].get("mean_W_run") else "")
                                 + (f", report in <code>{p['dir']}</code>" if p else ", not recorded") + "</p>")
                        yield keep, _stats_html(ev["stats"], extra), (p.get("mp4") if p and os.path.isfile(p.get("mp4", "")) else None), files or None, s, "\n".join(lines[-15:])
                    else:
                        yield ev["frame"], _stats_html(ev["stats"]), keep, keep, keep, "\n".join(lines[-15:])
            except Exception as e:                              # noqa: BLE001
                log(f"ERROR {type(e).__name__}: {e}")
                yield keep, _stats_html(None), keep, keep, keep, "\n".join(lines[-15:])
                raise gr.Error(str(e))
        finally:
            RUN_LOCK.release()

    with gr.Blocks(title="RipBench edge app", delete_cache=(300, 300)) as demo:
        gr.Markdown("## Rip-current edge pipeline — classification gate → light detector at a duty cycle\n"
                    "Companion app to the ICRA 2027 edge paper. Pick a source and a model, press **Start**; the overlay shows the gate "
                    "decision, detections, detector duty, processing fps and (on a Jetson) the board power. **Record** exports an "
                    "annotated mp4 + per-frame TSV + summary json. Thresholds default to the values frozen on the validation split.")
        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                with gr.Row():
                    start = gr.Button("▶ Start", variant="primary", elem_id="btn_start")
                    stop = gr.Button("■ Stop", variant="stop", elem_id="btn_stop")
                src_kind = gr.Radio(kinds, value="Sample clip", label="Source", elem_id="src_kind")
                sample = gr.Dropdown(choices=samples, value=samples[0][1] if samples else None, label="Sample clip", visible=True, elem_id="sample_clip")
                upload = gr.Video(sources=["upload"], label="Video file", include_audio=True, visible=False)
                url = gr.Textbox(label="RTSP / HTTP URL", placeholder="rtsp://user:pass@camera:554/stream  or  http://host/cam.mjpg", visible=False)
                cam = gr.Number(value=0, precision=0, label="Webcam index (/dev/videoN on the machine running the app)", visible=False)
                model = gr.Dropdown(choices=choices, value=stem0, label="Detector (from results/export_manifest.tsv)")
                imgsz = gr.Radio(choices=sizes0, value=sz0, label="Input size (px)")
                conf = gr.Slider(0.0, 1.0, value=conf_of(stem0, sz0)[0], step=0.01, label="Detector conf threshold")
                conf_info = gr.Markdown(conf_of(stem0, sz0)[1])
                reset = gr.Button("Reset conf to the val-frozen value", size="sm")
                with gr.Row():
                    gate_on = gr.Checkbox(value=True, label="Classification gate (yolo26n-cls 224)")
                    gate_thr = gr.Slider(0.0, 1.0, value=GATE_THR, step=0.01, label="Gate threshold P(rip)")
                duty = gr.Slider(0.2, 10.0, value=DEFAULT_DUTY_FPS, step=0.1, label="Duty cycle (decisions per second of video)")
                record = gr.Checkbox(value=True, label="Record report (annotated mp4 + per-frame TSV + summary json)")
                with gr.Accordion("Advanced", open=False):
                    backend = gr.Dropdown(["auto", "cpu", "cuda", "trt-engine"], value=DEFAULT_BACKEND, label="Backend",
                                          info="auto = TensorRT engine when one exists for the model, else onnxruntime CPU")
                    pace = gr.Checkbox(value=True, label="Real-time pacing for video files (off = as fast as possible)")
                    max_s = gr.Number(value=0, label="Max duration (s, 0 = whole source)")
            with gr.Column(scale=3):
                view = gr.Image(label="Live overlay", elem_id="live_view", type="numpy", format="jpeg", interactive=False, height=680, buttons=["fullscreen"])
                stats = gr.HTML(_stats_html(None), elem_id="stats_html")
                with gr.Accordion("Report of the last run", open=False):
                    out_video = gr.Video(label="Annotated video", interactive=False)
                    out_files = gr.File(label="Files (mp4, per-frame tsv, summary json)", file_count="multiple", interactive=False)
                    out_json = gr.JSON(label="summary.json")
                logbox = gr.Textbox(label="Log", lines=8, interactive=False)
        src_kind.change(on_kind, src_kind, [sample, upload, url, cam])
        model.change(on_model, [model, imgsz], [imgsz, conf, conf_info])
        imgsz.change(on_size, [model, imgsz], [conf, conf_info])
        reset.click(on_size, [model, imgsz], [conf, conf_info])
        start.click(ui_run, [src_kind, sample, upload, url, cam, model, imgsz, backend, duty, gate_on, gate_thr, conf, record, pace, max_s],
                    [view, stats, out_video, out_files, out_json, logbox])
        stop.click(ui_stop, None, logbox, queue=False)
    return demo


def main():
    if MODE == "headless":
        jobs = HEADLESS_JOBS
        if os.environ.get("RIPAPP_JOBS"):
            with open(os.environ["RIPAPP_JOBS"]) as f:
                jobs = json.load(f)
        run_headless(jobs)
        return
    import gradio as gr
    reg = load_registry()
    if not reg:
        sys.exit(f"no ONNX detectors found (manifest {MANIFEST_TSV}, exports {EXPORTS_DIR})")
    os.makedirs(OUT_ROOT, exist_ok=True)
    demo = build_ui(reg)
    print(f"RipBench edge app {APP_VERSION}: {len(reg)} detectors, gate {os.path.basename(GATE_ONNX)}, reports -> {OUT_ROOT}", flush=True)
    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, allowed_paths=[OUT_ROOT], show_error=True, inbrowser=False,
                theme=gr.themes.Soft(), css=".gradio-container{max-width:1600px !important}")


if __name__ == "__main__":
    main()
