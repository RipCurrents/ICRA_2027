#!/usr/bin/env python3
"""Event-level deployment metrics for the ICRA edge pipeline (test split, offline from dumps).

A beach camera is judged by WHEN it raises an alarm on a rip video and HOW OFTEN it alarms
wrongly on no-rip video, not by per-frame F2. This script replays every test video as a stream
from the existing prediction dumps (sampled frames = every 6th native frame, i.e. ~5 fps) at a
DUTY rate (frames kept per second), optionally behind the classification gate (yolo26n-cls 224
probabilities, same dumps as cascade_simulation.py), and applies the detector at its VAL-FROZEN
conf (selected exactly like the benchmark evaluator through bootstrap_cis.select_conf).
  frame positive  := detector ran (gate open) and >= 1 box with score >= conf
  alarm           := k-of-n persistence over the kept-frame sequence (KOFN)
  rip videos      : latency = t(first alarm) - t(first expert-annotated rip frame), s; also the
                    fraction of rip videos ever alarmed
  no-rip videos   : false-alarm episodes per hour (positives merged while gaps < GAP_S)
  no-rip videos   : also the fraction of no-rip videos with >= 1 false episode (robust: only ~0.3 h of
                    no-rip test footage exists, so episodes/hour rest on few clips)
Two operating points per stem: the FRAME-level conf (val-swept F2@50, as in every table of the paper) and
an EVENT-level conf selected on VAL as the conf maximising the video-level F2 of the alarm decision
(rip video alarmed = TP, no-rip video alarmed = FP, rip video silent = FN) at the reference setting
EVENT_SEL_SETTING (gate off, 1 fps, 1/1); both are then frozen and replayed on TEST.
Per-video fps comes from ffprobe of the raw videos (cached in manifests/{val,test}_video_fps.tsv).
Rows -> results/event_metrics.tsv (one per stem x duty x gate x kofn, plus seed-mean rows).
Constants below, no CLI args. Env: conda yoloV26 (CPU only).
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bootstrap_cis as bc  # noqa: E402

TASK = "bbox_from_instance"
VARIANTS = {"yolo26n canonical": r"^canon_bfi_n_s\d+_best_e\d+$",
            "yolo26n dense": r"^t1_nano_k025_t024_s1_n_s\d+_best_e\d+$",
            "yolo26s dense": r"^t1_nano_k025_t024_s1_s_s\d+_best_e\d+$"}
CLS_JSON = os.path.join(bc.DATASET_ROOT, "models", "classification", "best_models", "predictions",
                        "3_cls_yolo26_nano_224_best_e50_j2856380_preds_test.json")
GATE = [None, 0.50]              # None = detector always on; 0.50 = the val-selected gate threshold (cascade_sim.tsv)
DUTY_FPS = [5.0, 1.0, 0.5, 0.2]  # kept frames per second (5.0 = every sampled frame)
KOFN = [(1, 1), (2, 3)]
GAP_S = 10.0                     # no-rip positives closer than this merge into one false-alarm episode
VIDEO_DIRS = ["/home/user/RipBench/Videos/Rips", "/home/user/RipBench/Videos/No-Rips"]
FPS_CACHE = os.path.join(HERE, "manifests", "{split}_video_fps.tsv")
EVENT_SEL_SETTING = (None, 1.0, 1, 1)   # (gate, duty_fps, k, n) used to select the event-level conf on val
BETA = 2.0
OUT_TSV = os.path.join(HERE, "results", "event_metrics.tsv")
VIDEO_RE = re.compile(r"(RipBench-(?:NR-)?\d+)_(\d+)\.jpg$")


def video_fps(videos, split):
    path = FPS_CACHE.format(split=split)
    cache = {}
    if os.path.isfile(path):
        cache = {r["video"]: float(r["fps"]) for r in csv.DictReader(open(path), delimiter="\t")}
    for v in videos:
        if v in cache:
            continue
        files = [p for d in VIDEO_DIRS for p in glob.glob(os.path.join(d, v + ".*")) + glob.glob(os.path.join(d, v + "_*"))]
        fps = 30.0
        if files:
            r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                                "-of", "csv=p=0", files[0]], capture_output=True, text=True).stdout.strip()
            num, den = r.split("/") if "/" in r else (r, "1")
            fps = float(num) / float(den)
        else:
            print(f"WARNING: no raw video for {v}; assuming 30 fps", flush=True)
        cache[v] = fps
    with open(path, "w") as f:
        f.write("video\tfps\n")
        for v in sorted(cache):
            f.write(f"{v}\t{cache[v]:.4f}\n")
    return cache


def replay(frames, positives, fps, duty, gate_open, k, n):
    """frames: sorted list of (native_idx, image_id). Returns list of alarm times (s) after persistence."""
    step = max(1, int(round((fps / 6.0) / duty)))
    kept = frames[::step]
    flags, times = [], []
    for idx, iid in kept:
        flags.append(bool(gate_open.get(iid, True) and positives.get(iid, False)))
        times.append(idx / fps)
    alarms = []
    for i in range(len(kept)):
        w = flags[max(0, i - n + 1):i + 1]
        if sum(w) >= min(k, len(w)) if k > 1 else flags[i]:
            alarms.append(times[i])
    return alarms, len(kept)


def prep(gt, split):
    frames, first_gt, by_name = {}, {}, {}
    for im in gt.dataset["images"]:
        v, idx = VIDEO_RE.search(im["file_name"]).groups()
        frames.setdefault(v, []).append((int(idx), im["id"]))
        by_name[im["id"]] = os.path.basename(im["file_name"])
        if gt.getAnnIds(imgIds=[im["id"]]):
            first_gt[v] = min(first_gt.get(v, 10 ** 9), int(idx))
    for v in frames:
        frames[v].sort()
    fps = video_fps(sorted(frames), split)
    cls = json.load(open(CLS_JSON.replace("_preds_test", f"_preds_{split}")))
    cls = {iid: cls[by_name[iid]] for iid in by_name}
    return frames, first_gt, fps, cls


def evaluate(frames, first_gt, fps, cls, positives, gate, duty, k, n):
    rip_videos = [v for v in frames if "NR-" not in v]
    nr_videos = [v for v in frames if "NR-" in v]
    nr_hours = sum((frames[v][-1][0] + 1) / fps[v] for v in nr_videos) / 3600.0
    gate_open = {} if gate is None else {iid: (pr >= gate) for iid, pr in cls.items()}
    lat, hit, fa, fa_videos, kept_tot = [], 0, 0, 0, 0
    for v in rip_videos:
        alarms, kept = replay(frames[v], positives, fps[v], duty, gate_open, k, n)
        kept_tot += kept
        t0 = first_gt.get(v, frames[v][0][0]) / fps[v]
        after = [a for a in alarms if a >= t0]
        if after:
            hit += 1
            lat.append(after[0] - t0)
    for v in nr_videos:
        alarms, kept = replay(frames[v], positives, fps[v], duty, gate_open, k, n)
        kept_tot += kept
        episodes, last = 0, -1e9
        for a in alarms:
            if a - last >= GAP_S:
                episodes += 1
            last = a
        fa += episodes
        fa_videos += episodes > 0
    n_frames_all = sum(len(frames[v]) for v in frames)
    det_runs = sum(1 for v in frames for idx, iid in frames[v][::max(1, int(round((fps[v] / 6.0) / duty)))]
                   if gate_open.get(iid, True))
    tp, fp, fn = hit, fa_videos, len(rip_videos) - hit
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fbeta = (1 + BETA ** 2) * prec * rec / (BETA ** 2 * prec + rec) if (BETA ** 2 * prec + rec) else 0.0
    return dict(rip_videos=len(rip_videos), rip_alarmed_frac=f"{hit / len(rip_videos):.3f}",
                latency_median_s=f"{np.median(lat):.1f}" if lat else "",
                latency_p90_s=f"{np.percentile(lat, 90):.1f}" if lat else "",
                norip_videos=len(nr_videos), norip_alarmed_frac=f"{fa_videos / len(nr_videos):.3f}",
                norip_hours=f"{nr_hours:.2f}", false_alarms_per_hour=f"{fa / nr_hours:.2f}",
                video_F2=f"{fbeta:.3f}",
                frames_kept_frac=f"{kept_tot / n_frames_all:.3f}", detector_runs_frac=f"{det_runs / n_frames_all:.3f}")


def main():
    ed = bc.load_ed()
    base = os.path.join(bc.DATASET_ROOT, "labels", TASK, "coco")
    gt_v, gt_t = ed.COCO(os.path.join(base, "val.json")), ed.COCO(os.path.join(base, "test.json"))
    V, T = prep(gt_v, "val"), prep(gt_t, "test")
    stems = []
    for name, rx in VARIANTS.items():
        for d in bc.PRED_DIRS:
            for p in glob.glob(os.path.join(d, "*_preds_test.json")):
                s = os.path.basename(p)[:-len("_preds_test.json")]
                if re.match(rx, s) and os.path.isfile(os.path.join(d, f"{s}_preds_val.json")):
                    stems.append((name, s, d))
    stems = sorted(set(stems))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for name, stem, d in stems:
        pv = json.load(open(os.path.join(d, f"{stem}_preds_val.json")))
        pt = json.load(open(os.path.join(d, f"{stem}_preds_test.json")))
        conf_frame = bc.select_conf(ed, gt_v, pv)
        # event-level operating point on VAL: video-level F2 of the alarm decision at the reference setting
        gate, duty, k, n = EVENT_SEL_SETTING
        best = None
        for c in ed.CONF_SWEEP:
            pos_v = {p["image_id"]: True for p in pv if p["score"] >= c}
            m = evaluate(*V, pos_v, gate, duty, k, n)
            key = (float(m["video_F2"]), float(m["rip_alarmed_frac"]), -c)
            if best is None or key > best[0]:
                best = (key, c)
        conf_event = best[1]
        for op, conf in (("frame-F2", conf_frame), ("event-F2", conf_event)):
            positives = {p["image_id"]: True for p in pt if p["score"] >= conf}
            for gate in GATE:
                for duty in DUTY_FPS:
                    for k, n in KOFN:
                        m = evaluate(*T, positives, gate, duty, k, n)
                        rows.append(dict(timestamp=ts, variant=name, stem=stem, operating_point=op, conf=f"{conf:.2f}",
                                         gate="off" if gate is None else f"{gate:.2f}", duty_fps=duty, k=k, n=n, **m))
        print(f"{stem}: frame conf {conf_frame:.2f} | event conf {conf_event:.2f}", flush=True)
    keys = ["variant", "operating_point", "gate", "duty_fps", "k", "n"]
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    for g, rs in groups.items():
        m = dict(rs[0])
        m["stem"] = f"MEAN over {len(rs)} seeds"
        m["conf"] = "/".join(r["conf"] for r in rs)
        for f in ("rip_alarmed_frac", "latency_median_s", "latency_p90_s", "norip_alarmed_frac", "false_alarms_per_hour",
                  "video_F2", "frames_kept_frac", "detector_runs_frac"):
            vals = [float(r[f]) for r in rs if r[f] != ""]
            m[f] = f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else ""
        rows.append(m)
    with open(OUT_TSV, "w") as f:
        f.write("\t".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r.values()) + "\n")
    print("wrote", OUT_TSV, len(rows), "rows")


if __name__ == "__main__":
    main()
