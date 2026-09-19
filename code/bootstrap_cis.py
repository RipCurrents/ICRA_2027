#!/usr/bin/env python3
"""Video-level bootstrap CIs + paired deltas for the ICRA student rows.

For every prediction dump in PRED_DIRS (<stem>_preds_{val,test}.json, the
benchmark predictor's COCO-results format) and every GT view in TASKS: select
the conf on val exactly like eval/evaluate_detection.py (imported, unmodified:
best F2@50, ties R/P/conf), freeze it, then on test compute PER-VIDEO TP/FP/FN
at IoU 0.50 (same greedy matcher) and bootstrap the 71 test videos (B draws,
seed 0) -> 95% percentile CI of pooled F2@50 (and AP50 is NOT bootstrapped
here - it needs pycocotools per draw; F2 is the paper's headline). PAIRS lists
(model_a, model_b) whose per-video counts are resampled jointly -> CI of the
F2 delta (paired, same videos). Rows -> results/student_bootstrap_cis.tsv.
Constants below, no CLI args. Env: conda yoloV26.
"""
import importlib.util
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
PRED_DIRS = [os.path.join(HERE, "predictions", "students"),
             "/mnt/linux/icra_edge/cluster_pull/predictions"]
TASKS = ("bbox_from_instance", "bbox")
STEM_FILTER = None          # substring filter on stems, e.g. "_n_s"; None = all
PAIRS = [                   # (model_a, model_b) stems -> CI of F2(b) - F2(a); missing stems skipped
]
GROUP_PAIRS = [
    (("h50_sparse_s", ['h50_sparse_s_s1_best_e9', 'h50_sparse_s_s2_best_e15']), ("h50_interp_s", ['h50_interp_s_s1_best_e2', 'h50_interp_s_s2_best_e3'])),
]  # ((name_a, [stems_a...]), (name_b, [stems_b...])): per-video TP/FP/FN POOLED over each variant's seeds
   # (micro-average), then one video-level bootstrap of the delta
B, SEED = 1000, 0
OUT_TSV = os.path.join(HERE, "results", "student_bootstrap_cis.tsv")
VIDEO_RE = re.compile(r"(RipBench-(?:NR-)?\d+)_\d+\.jpg$")


def load_ed():
    spec = importlib.util.spec_from_file_location("ed_bs", os.path.join(REPO_ROOT, "eval", "evaluate_detection.py"))
    ed = importlib.util.module_from_spec(spec)
    sys.modules["ed_bs"] = ed
    spec.loader.exec_module(ed)
    return ed


def per_video_counts(ed, gt, preds, conf):
    """{video: [TP, FP, FN]} at IoU .50 and the given conf (greedy matcher from the evaluator)."""
    pred_coco = gt.loadRes(preds) if preds else None
    img_video = {im["id"]: VIDEO_RE.search(im["file_name"]).group(1) for im in gt.dataset["images"]}
    out = {}
    for img_id, per_cat in zip(gt.getImgIds(), ed.build_cache(gt, pred_coco, "bbox")):
        v = img_video[img_id]
        c = out.setdefault(v, [0, 0, 0])
        for scores, ious, n_gt in per_cat.values():
            k = int((scores >= conf).sum())
            tp, fp, fn = ed.greedy_match(ious[:k], 0.50) if k or n_gt else (0, 0, 0)
            c[0] += tp
            c[1] += fp
            c[2] += fn
    return out


def f2_from(counts, idx, videos):
    tp = sum(counts[videos[i]][0] for i in idx)
    fp = sum(counts[videos[i]][1] for i in idx)
    fn = sum(counts[videos[i]][2] for i in idx)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0


def select_conf(ed, gt_v, preds_v):
    cache = ed.build_cache(gt_v, gt_v.loadRes(preds_v) if preds_v else None, "bbox")
    best = max(((c, ed.metrics_at(cache, c, 0.50)) for c in ed.CONF_SWEEP),
               key=lambda s: (s[1]["F2"], s[1]["R"], s[1]["P"], s[0]))
    return best[0]


def main():
    ed = load_ed()
    rng = np.random.default_rng(SEED)
    stems = {}
    for d in PRED_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith("_preds_test.json"):
                s = f[:-len("_preds_test.json")]
                if STEM_FILTER and STEM_FILTER not in s:
                    continue
                if os.path.isfile(os.path.join(d, s + "_preds_val.json")):
                    stems[s] = d
    print(f"{len(stems)} models")
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    write_header = not os.path.isfile(OUT_TSV)
    out = open(OUT_TSV, "a")
    if write_header:
        out.write("timestamp\tmodel\ttask\tconf\tF2_test\tCI_lo\tCI_hi\tn_videos\tB\tpair_vs\tdelta\tdelta_lo\tdelta_hi\n")
    for task in TASKS:
        gt_v = ed.COCO(os.path.join(DATASET_ROOT, "labels", task, "coco", "val.json"))
        gt_t = ed.COCO(os.path.join(DATASET_ROOT, "labels", task, "coco", "test.json"))
        counts, point = {}, {}
        for s, d in sorted(stems.items()):
            preds_v = json.load(open(os.path.join(d, s + "_preds_val.json")))
            preds_t = json.load(open(os.path.join(d, s + "_preds_test.json")))
            conf = select_conf(ed, gt_v, preds_v)
            c = per_video_counts(ed, gt_t, preds_t, conf)
            videos = sorted(c)
            all_idx = np.arange(len(videos))
            f2 = f2_from(c, all_idx, videos)
            draws = np.array([f2_from(c, rng.integers(0, len(videos), len(videos)), videos) for _ in range(B)])
            lo, hi = np.percentile(draws, [2.5, 97.5])
            counts[s], point[s] = (c, videos), (conf, f2)
            out.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{s}\t{task}\t{conf:.2f}\t{f2:.4f}\t{lo:.4f}\t{hi:.4f}\t{len(videos)}\t{B}\t\t\t\t\n")
            print(f"[{task}] {s}: conf {conf:.2f} F2 {f2:.4f} [{lo:.4f}, {hi:.4f}] n={len(videos)}")
        for (na, sa), (nb, sb) in GROUP_PAIRS:
            if not all(x in counts for x in sa + sb):
                print(f"[{task}] GROUP_PAIRS {na} vs {nb}: missing stems, skipped")
                continue
            videos = counts[sa[0]][1]
            def pooled(stems):
                out = {v: [0, 0, 0] for v in videos}
                for st in stems:
                    for v, c in counts[st][0].items():
                        for k in range(3):
                            out[v][k] += c[k]
                return out
            ca, cb = pooled(sa), pooled(sb)
            rng_g = np.random.default_rng(SEED)
            all_idx = np.arange(len(videos))
            fa, fb = f2_from(ca, all_idx, videos), f2_from(cb, all_idx, videos)
            deltas = []
            for _ in range(B):
                idx = rng_g.integers(0, len(videos), len(videos))
                deltas.append(f2_from(cb, idx, videos) - f2_from(ca, idx, videos))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            out.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{nb}[pooled {len(sb)} seeds]\t{task}\t\t{fb:.4f}\t\t\t{len(videos)}\t{B}\t{na}[pooled {len(sa)} seeds]\t{fb - fa:.4f}\t{lo:.4f}\t{hi:.4f}\n")
            print(f"[{task}] POOLED {nb} ({len(sb)} seeds, F2 {fb:.4f}) - {na} ({len(sa)} seeds, F2 {fa:.4f}): dF2 {fb - fa:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        for a, b in PAIRS:
            if a not in counts or b not in counts:
                continue
            (ca, videos), (cb, _) = counts[a], counts[b]
            rng_p = np.random.default_rng(SEED)
            deltas = []
            for _ in range(B):
                idx = rng_p.integers(0, len(videos), len(videos))
                deltas.append(f2_from(cb, idx, videos) - f2_from(ca, idx, videos))
            d0 = point[b][1] - point[a][1]
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            out.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{b}\t{task}\t{point[b][0]:.2f}\t{point[b][1]:.4f}\t\t\t{len(videos)}\t{B}\t{a}\t{d0:.4f}\t{lo:.4f}\t{hi:.4f}\n")
            print(f"[{task}] {b} - {a}: dF2 {d0:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    out.close()
    print(f"-> {OUT_TSV}")


if __name__ == "__main__":
    main()
