#!/usr/bin/env python3
"""FULL-SYSTEM reference row: live camera -> cls gate -> duty-cycled detector on a board (RPi 4 /
Jetson). Complements the ladder (which ISOLATES model cost on stored frames): this measures the whole
deployed pipeline, gross wall watts, camera included and named. Run one window per scenario, e.g.
camera at a monitor playing (a) a no-rip beach clip (low duty) and (b) a rip-heavy clip (high duty)
-> brackets the deployment power envelope. NEVER use camera-off-a-monitor output for accuracy numbers
(moire/exposure/color shift = not the benchmark distribution); accuracy rows come from the stored-
frame dumps only.

Loop: capture continuously; every GATE_PERIOD_S run the gate at 224; while the most recent gate says
rip (P >= GATE_THR), run the detector on captured frames at DET_HZ. Stats + optional Shelly power
sidecar -> one TSV row per window. Wall meter read per measurement_protocol.md during the window
(or SHELLY_IP for automatic 1 Hz logging once the plug arrives). Constants below, no CLI args.
Needs: numpy, opencv-python, onnxruntime (same venv as board_bench); board_bench.py in this folder.
"""
import json
import os
import platform
import threading
import time
from datetime import datetime

import numpy as np

# --- CONFIG -------------------------------------------------------------------
SOURCE = 0                    # int = /dev/video<N> (UVC webcam: zero friction). Pi CSI camera on
                              # libcamera OSes: set to a GStreamer string, e.g.
                              # "libcamerasrc ! video/x-raw,width=1280,height=720 ! videoconvert ! appsink"
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30   # requested; actuals are recorded
CAMERA_NAME = "SET_ME"        # e.g. "Pi Camera Module v2" / "Logitech C920" - goes in the row verbatim
BOARD = platform.node()
GATE_ONNX = "device_kit/models/yolo26n_cls_224_gate.onnx"
DET_ONNX = "device_kit/models/t1_nano_k025_t024_s1_n_s1_best_e2_384.onnx"   # deployment model+size
DET_IMGSZ = 384
DET_CONF = 0.04               # VAL-frozen (operating_points.tsv; bfi row of the chosen model)
GATE_THR = 0.50
GATE_PERIOD_S = 1.0           # gate cadence (1 Hz)
DET_HZ = 5.0                  # detector rate while the gate is positive
CLASSIFY_ONLY = False         # True = scenario S2 (lifeguard alert): gate only, alarm iff P(rip) >= GATE_THR; detector never runs
WINDOW_S = 600                # >= 10 min per protocol
SCENARIO = "SET_ME"           # e.g. "monitor_norip_clip_RipBench-NR-042" / "monitor_rip_clip_113830"
THREADS = 4
SHELLY_IP = ""                # e.g. "192.168.0.60" -> poll /rpc/Switch.GetStatus?id=0 at 1 Hz into the sidecar
OUT_TSV = "ICRA_2027/device_results/pipeline_power.tsv"
POWER_SIDECAR_DIR = "ICRA_2027/device_results/shelly"
# ------------------------------------------------------------------------------


def shelly_poll(stop, rows):
    import urllib.request
    while not stop.is_set():
        try:
            with urllib.request.urlopen(f"http://{SHELLY_IP}/rpc/Switch.GetStatus?id=0", timeout=2) as r:
                rows.append((time.time(), json.load(r).get("apower")))
        except Exception:
            rows.append((time.time(), None))
        stop.wait(1.0)


def main():
    import cv2
    import onnxruntime as ort
    import board_bench as bb
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    gate = ort.InferenceSession(GATE_ONNX, so, providers=["CPUExecutionProvider"])
    gname = gate.get_inputs()[0].name
    det = dname = None
    if not CLASSIFY_ONLY:                     # S2 never loads the detector (device review item 1)
        det = ort.InferenceSession(DET_ONNX, so, providers=bb.PROVIDERS)
        dname = det.get_inputs()[0].name
    cap = cv2.VideoCapture(SOURCE) if isinstance(SOURCE, int) else cv2.VideoCapture(SOURCE, cv2.CAP_GSTREAMER)
    assert cap.isOpened(), f"camera source {SOURCE!r} failed to open"
    if isinstance(SOURCE, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    ok, frame = cap.read()
    assert ok, "no frame from camera"
    actual = f"{frame.shape[1]}x{frame.shape[0]}"
    print(f"{BOARD} | camera {CAMERA_NAME} {actual} | gate {GATE_PERIOD_S}s | det {DET_HZ}Hz while positive | window {WINDOW_S}s")
    stop, power_rows = threading.Event(), []
    if SHELLY_IP:
        threading.Thread(target=shelly_poll, args=(stop, power_rows), daemon=True).start()
    temp0 = bb.read_temp()
    n_frames = n_gate = n_gate_pos = n_det = n_det_hits = 0
    gate_positive = False
    last_gate = last_det = 0.0
    t_start = time.time(); t_end = t_start + WINDOW_S
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            print("  dropped frame", flush=True)
            continue
        n_frames += 1
        now = time.time()
        if now - last_gate >= GATE_PERIOD_S:
            x, *_ = bb.classify_preprocess(frame, 224)   # v2: training transform (center crop), not letterbox
            probs = gate.run(None, {gname: x})[0][0]
            p_rip = float(probs[1]) if len(probs) == 2 else float(probs.max())
            gate_positive = p_rip >= GATE_THR
            n_gate += 1
            n_gate_pos += int(gate_positive)
            last_gate = now
        if gate_positive and not CLASSIFY_ONLY and now - last_det >= 1.0 / DET_HZ:
            x, r, top, left = bb.letterbox(frame, DET_IMGSZ)
            boxes, scores = bb.decode(det.run(None, {dname: x}), r, top, left, DET_CONF)
            n_det += 1
            n_det_hits += int(len(scores) > 0)
            last_det = now
    stop.set()
    cap.release()
    dur = time.time() - t_start                # actual elapsed, not the nominal window
    watts = [w for _, w in power_rows if w is not None]
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    hdr = not os.path.isfile(OUT_TSV)
    with open(OUT_TSV, "a") as f:
        if hdr:
            f.write("timestamp\tboard\tcamera\tres\tscenario\tgate_model\tgate_thr\tclassify_only\tdet_model\tdet_imgsz\tdet_conf\twindow_nominal_s\twindow_actual_s"
                    "\tcapture_fps\tgate_per_s\tgate_pos_rate\tdet_per_s\tdet_hit_rate\ttemp_start_C\ttemp_end_C"
                    "\tthrottled\tshelly_W_mean\tshelly_W_min\tshelly_W_max\tshelly_n\twall_meter_note\n")
        shelly = (f"{np.mean(watts):.2f}\t{np.min(watts):.2f}\t{np.max(watts):.2f}\t{len(watts)}"
                  if watts else "-\t-\t-\t0")
        f.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{BOARD}\t{CAMERA_NAME}\t{actual}\t{SCENARIO}"
                f"\t{os.path.basename(GATE_ONNX)}\t{GATE_THR}\t{int(CLASSIFY_ONLY)}\t{'-' if CLASSIFY_ONLY else os.path.basename(DET_ONNX)}\t{DET_IMGSZ if not CLASSIFY_ONLY else '-'}\t{DET_CONF if not CLASSIFY_ONLY else '-'}\t{WINDOW_S}\t{dur:.1f}"
                f"\t{n_frames / dur:.1f}\t{n_gate / dur:.2f}\t{n_gate_pos / max(n_gate, 1):.3f}"
                f"\t{n_det / dur:.2f}\t{n_det_hits / max(n_det, 1):.3f}\t{temp0:.1f}\t{bb.read_temp():.1f}"
                f"\t{bb.throttled_flags()}\t{shelly}\tZX-1494 readings in device_wall_meter.tsv\n")
    if power_rows:
        os.makedirs(POWER_SIDECAR_DIR, exist_ok=True)
        side = os.path.join(POWER_SIDECAR_DIR, f"{BOARD}_{SCENARIO}_{datetime.now():%Y%m%d_%H%M}.tsv")
        with open(side, "w") as f:
            f.write("unix_ts\tapower_W\n")
            for ts, w in power_rows:
                f.write(f"{ts:.1f}\t{w if w is not None else 'NA'}\n")
        print(f"shelly sidecar -> {side}")
    print(f"frames {n_frames} ({n_frames / dur:.1f}/s) | gate pos rate {n_gate_pos / max(n_gate, 1):.2f} | "
          f"det {n_det / dur:.2f}/s, hit rate {n_det_hits / max(n_det, 1):.2f}")


if __name__ == "__main__":
    main()
