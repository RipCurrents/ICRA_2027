# app/ — rip-current edge app (classifier gate → light detector)

A single-purpose deployment / visualisation app for the paper's edge pipeline: **classification gate
(yolo26n-cls 224) → light yolo26 detector at a duty cycle**, run on a video file, an RTSP/HTTP stream or a
webcam, with a live overlay (boxes + scores, gate state, detector duty, processing fps and — on a Jetson — the
board power from `tegrastats`) and an exportable report (annotated mp4 + per-frame TSV + summary json).
The gate input uses the same centre-crop transform as the board runtime (`code/board_bench.py`). It is **not** a research
tool (no threshold tuning, no evaluation — the paper's numbers come from the frozen result ledgers).

Files: `app.py` (pipeline + headless mode + Gradio UI, one file), `requirements.txt`, `run.sh`, this README. Reused, never modified: `code/board_bench.py` (letterbox + ONNX decode,
validated against the benchmark evaluator within ±0.005 F2), `code/hil_demo.py` (gate softmax handling, copied),
`code/jetson_measure.py` (`Tegrastats` reader, `TrtRunner`).

## Quick start
```bash
# once: an isolated env (never into yoloV26 / rfdetr / maskdino)
conda create -n icra_app python=3.11 pip -y && ~/anaconda3/envs/icra_app/bin/pip install -r requirements.txt
# UI (http://<host>:7860; the machine running the app reads the webcam / stream, not the browser)
./run.sh
# headless batch (HEADLESS_JOBS in app.py, or a json list of jobs)
./run.sh headless [jobs.json]
# long-running server, house hygiene (two kernel OOM kills happened on this PC):
systemd-run --user --collect --unit=app-ui -p MemoryMax=8G -p WorkingDirectory=$PWD bash -c './run.sh > /mnt/linux/icra_edge/app_runs/app-ui.log 2>&1'
```
Everything is a constant at the top of `app.py` (house rules: no CLI args). The only environment overrides are
the ones `run.sh` needs to switch machines without editing the file: `RIPAPP_MODE` (ui|headless), `RIPAPP_JOBS`
(json job list), `RIPAPP_PORT`, `RIPAPP_EXPORTS_DIR`, `RIPAPP_OUT_ROOT`, `RIPAPP_ENGINE_DIR`, `RIPAPP_MANIFEST`,
`RIPAPP_EVAL_TSV`, `RIPAPP_PYTHON`.

## What the UI does
- **Sources**: sample clip (the held-out TEST demo videos: RipBench-038 beach cam, 025 drone, 125 failure case,
  the NR-021 + 038 deployment replay), uploaded video file, RTSP/HTTP URL, webcam index (server side, `/dev/videoN`).
  Files are replayed frame by frame with one decision every `round(fps / duty)` frames (identical to
  `hil_demo.py`), optionally paced to real time. Webcam / streams are *live*: a reader thread keeps only the newest
  frame (a slow board drops frames instead of accumulating latency; `frames_grabbed` vs `frames_processed` in
  the summary shows how many), decisions are scheduled on the wall clock, the loop runs at `LIVE_LOOP_FPS` (10).
  RTSP transport is negotiated by ffmpeg (UDP, TCP fallback); set `RTSP_TRANSPORT = "tcp"` for lossy WiFi links /
  cameras behind NAT (note: some servers, e.g. VLC's RTSP output, refuse TCP-interleaved with a 461).
  Tested sources: local mp4 files, webcam `/dev/video0`, MPEG-TS over HTTP (ffmpeg `-listen 1`), RTSP (VLC
  `#rtp{sdp=rtsp://...}`). Python's `http.server` cannot serve an mp4 to ffmpeg (no HTTP range support).
- **Controls**: detector (dropdown built from `train/edge_distill/results/export_manifest.tsv`, paper students first, every
  exported seed/ablation after), input size (640/512/384, whichever are exported), conf threshold (defaults to the
  **val-frozen** value of that model × size from `train/edge_distill/results/student_eval_bfi.tsv`, column `selected_conf`; a
  reset button restores it), gate on/off + threshold (default 0.50, val-selected — RESULTS.md cascade section),
  duty cycle (0.2–10 decisions per second of video; 1/s default), record on/off; *Advanced*: backend, real-time
  pacing, max duration.
- **Display**: overlay stream (≤ `DISPLAY_FPS` = 8 refreshes/s, resized to 1280 px wide for the browser; the
  recorded mp4 keeps every processed frame at native resolution), HUD (`HUD_SCALE` = 1.6 at 1080p, sized for a
  720p viewing of the demo; shrinks to fit portrait/low-res frames) with the model | input size | engine
  (e.g. `TensorRT FP16`), gate P(rip)/state/ms, detector conf / decisions run / duty / ms / count, processing fps,
  a power line, a "RIP CURRENT" banner while a box is active; a stats table; the last run's report (playable
  video, files, json) and an end-of-run summary line (frames, decisions, duty, rip decisions, false alarms,
  mJ/decision, average W).
- **Power / energy (Jetson)**: when `tegrastats` exists, the run starts with a `POWER_IDLE_SECONDS` = 3 s idle window
  (models loaded, nothing running) and a `POWER_BURST_SECONDS` = 2 s continuous-inference window → `idle_W` /
  `active_W` (the measurement protocol's two states in miniature; `jetson_measure.py` owns the real 200/1000-frame
  protocol). The HUD then shows the live VDD_IN power, idle/active, and **mJ per decision =
  (active_W − idle_W) × (gate ms + detector ms)**; the summary adds the run's mean W and Wh, and — for citing the run
  file in the paper's energy sentence (the internal critic's request) — `power.sample_interval_ms`, `power.rail`, the idle and
  active calibration windows (start/end timestamps, seconds, inference count, state) and the formula. Elsewhere "n/a".
  Test hook: `RIPAPP_POWER_SIM=1` fakes the rails (4 W + 3 W × CPU busy fraction) so the code path runs on this PC
  — every such output is labelled SIMULATED and must never be quoted.
- **False alarms**: demo clips carry a rip window (`RIP_WINDOWS`, seconds of video that contain a rip; the
  deployment replay = 20–50 s; the all-rip test videos = whole clip; headless jobs: key `rip_window`); rip
  decisions outside the window count as false alarms in the stats line and `summary.json` (`false_alarm_ticks`).
  Other sources: not counted (no ground truth).
- **Report** (`Record` on) → `OUT_ROOT/<timestamp>_<source>_<model>_<size>/`:
  `overlay.mp4` (H.264 via ffmpeg when available, else mp4v), `frames.tsv` (one row per processed frame:
  `frame time_s wall_time tick gate_p gate_open det_ran n_det n_det_raw gate_ms det_ms power_w mj_decision
  boxes_xyxy_score`; non-tick rows carry the boxes persisted from the last decision), `summary.json`
  (source, model + provenance incl. the frozen conf and benchmark test F2, backend, duty, ticks, detector runs
  and duty fraction, rip ticks, first-rip time, gate/detector ms mean & p90, processing fps, real-time factor,
  power rails mean W + Wh, output paths, git hash, host).
- **Heavy reference**: `HEAVY_MODELS` adds the ICRA session's validated RF-DETR-n detection ONNX (30 M params, 384 px
  only) to the dropdown as "RF-DETR-n · heavy reference", with its own pre/post-processing imported from
  `code/rfdetr_onnx_bench.py` (square resize + ImageNet norm, sigmoid top-300, rip = logit index 0,
  DETR → no NMS); val-frozen conf **0.26**, benchmark test F2@50 0.706 on the manual **tight** GT view (the students'
  numbers are on the bfi view — the UI labels the view next to every F2). This entry is the x86 reference configuration;
  the paper's deployed RF-DETR-n rows use the board runtime's INTER_AREA pre-downscale and their own validation-selected
  confidences (`results/device/`, `code/deployed_cascade_metrics.py`). ≈ 115 ms/decision at the app's 4 CPU
  threads (62 ms at 8), i.e. still fine at 1 decision/s.
- Detections: the exports are end-to-end (NMS-free head, output `(1, 300, 6)`); at the recall-heavy frozen conf
  they emit near-duplicate low-score boxes, so the app applies class-agnostic NMS at IoU 0.7 (`NMS_IOU`, the same
  0.7 the benchmark dumps are post-processed with) — `n_det_raw` in the TSV keeps the raw count.

## Headless mode (for batch demo videos)
`./run.sh headless jobs.json` — a json list of jobs; unknown keys fall back to `HEADLESS_DEFAULTS`:
```json
[{"source": "/home/user/RipBench/Videos/Rips/RipBench-038.mp4", "model": "t1_nano_k025_t024_s1_n_s1_best_e2",
  "imgsz": 384, "duty_fps": 1.0, "gate": true, "gate_thr": 0.5, "conf": null, "backend": "auto",
  "record": true, "pace_realtime": false, "max_seconds": 0, "out_name": "demo_038"},
 {"source_kind": "webcam", "source": 0, "max_seconds": 60},
 {"source_kind": "url", "source": "rtsp://cam/stream", "max_seconds": 120}]
```
`conf: null` = the val-frozen value. Progress lines every 5 s; the run ends with `RIPAPP_HEADLESS_DONE` and the
output paths. `run_pipeline(cfg)` is importable too (a generator; consume it to run).

## Backends
`auto` (default): a TensorRT engine `ENGINE_DIR/<onnx stem>_<ENGINE_PRECISION>.engine` (the naming of
`jetson_measure.build_engine` / `trtexec`) when it exists and `tensorrt`+`pycuda` import, else onnxruntime CPU
(`THREADS` = 4 intra-op threads, an edge-like budget on any host). `cpu`, `cuda` (onnxruntime-gpu; falls back to
CPU with a log line if the provider is missing), `trt-engine` (error if the engine is missing). The 4090 is
never touched unless `cuda` is chosen explicitly.

Measured on this PC (CPU, 4 threads, onnxruntime 1.29): gate 2–5 ms, yolo26n@384 7–12 ms, yolo26n@640 ≈ 23 ms,
yolo26s@384 ≈ 19 ms per inference; a 1080p file with recording processes at ≈ 60 fps (2× real time).
