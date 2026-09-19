#!/usr/bin/env python3
"""Instability-proof alarm operating point (the internal critic O2, 2026-09-07), VAL only: candidate confs are the coarse grid
restricted to those whose frame-level val F2 (tight-view GT, benchmark evaluator counts) is within 0.02 of the detector's
max frame F2; the alarm op replaces the frame-frozen op ONLY if val video-F2 gains >= 0.05 over the frame op's video-F2.
Output results/alarm_op_rule_val.tsv with the verdict per detector. Constants at top, no CLI args."""
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
GT_VAL_BFI = os.path.join(DATASET_ROOT, "labels", "bbox_from_instance", "coco", "val.json")
GT_VAL_TIGHT = os.path.join(DATASET_ROOT, "labels", "bbox", "coco", "val.json")
CLS_VAL = os.path.join(DATASET_ROOT, "models", "classification", "best_models", "predictions", "3_cls_yolo26_nano_224_best_e50_j2856380_preds_val.json")
PRED = os.path.join(DATASET_ROOT, "models", "bbox", "best_models", "predictions")
DETECTORS = {  # name -> (val dump, frame-frozen conf on the tight view)
    "yolo26n dense (deployed)": ("/mnt/linux/icra_edge/predictions/students/t1_nano_k025_t024_s1_n_s1_best_e2_preds_val.json", 0.04),
    "yolo26s dense": ("/mnt/linux/icra_edge/predictions/students/t1_nano_k025_t024_s1_s_s3_best_e3_preds_val.json", 0.02),
    "yolo26s expert-label": (None, None),
    "RF-DETR-n": (os.path.join(PRED, "1_det_rfdetr_nano_best_best_ema_j3178618_preds_val.json"), 0.23),
    "RF-DETR-s": (os.path.join(PRED, "3_det_rfdetr_small_best_best_ema_j3178622_preds_val.json"), 0.26),
    "RF-DETR-m": (os.path.join(PRED, "5_det_rfdetr_medium_best_best_ema_j3178623_preds_val.json"), 0.31),
}
GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
FRAME_TOL, VF2_GAIN = 0.02, 0.05
GATE, DUTY, K, N = 0.50, 1.0, 1, 1
OUT_TSV = os.path.join(HERE, "results", "alarm_op_rule_val.tsv")


def main():
    import csv
    import event_metrics as em
    from pycocotools.coco import COCO
    spec = importlib.util.spec_from_file_location("ed_r", os.path.join(REPO, "eval", "evaluate_detection.py"))
    ED = importlib.util.module_from_spec(spec); sys.modules["ed_r"] = ED; spec.loader.exec_module(ED)
    p = sorted(glob.glob("/mnt/linux/icra_edge/*/canon_bfi_s_s1_best_e1_preds_val.json") + glob.glob("/mnt/linux/icra_edge/cluster_pull/predictions/canon_bfi_s_s1_best_e1_preds_val.json"))[0]
    conf_s = next(float(r["selected_conf"]) for r in csv.DictReader(open(os.path.join(HERE, "results", "student_eval_tight.tsv")), delimiter="\t") if r["model"] == "canon_bfi_s_s1_best_e1" and r["split"] == "val")
    DETECTORS["yolo26s expert-label"] = (p, conf_s)
    gt_b = COCO(GT_VAL_BFI); gt_t = COCO(GT_VAL_TIGHT)
    frames, first_gt, fps, _ = em.prep(gt_b, "val")
    by_name = {im["id"]: os.path.basename(im["file_name"]) for im in gt_b.dataset["images"]}
    probs = json.load(open(CLS_VAL)); cls = {iid: float(probs[b]) for iid, b in by_name.items() if b in probs}
    out = open(OUT_TSV, "w"); out.write("detector\tframe_conf\tframe_vF2\tframe_F2max\talarm_conf\talarm_frameF2\talarm_vF2\tgain\tverdict\n")
    for name, (dump, fconf) in DETECTORS.items():
        preds = json.load(open(dump))
        cache = ED.build_cache(gt_t, gt_t.loadRes(preds), "bbox")
        frame_f2 = {c: ED.evaluate_split(cache, c)[0]["F2"] for c in sorted(set(GRID + [fconf]))}
        fmax = max(frame_f2.values())
        def vf2(c):
            pos = {p["image_id"]: True for p in preds if p["score"] >= c}
            return float(em.evaluate(frames, first_gt, fps, cls, pos, GATE, DUTY, K, N)["video_F2"])
        v_frame = vf2(fconf)
        elig = [c for c in GRID if frame_f2[c] >= fmax - FRAME_TOL]
        best = max(elig, key=lambda c: (vf2(c), c)); v_best = vf2(best)
        verdict = "ALARM OP" if v_best - v_frame >= VF2_GAIN else "KEEP FRAME OP"
        chosen = best if verdict == "ALARM OP" else fconf
        out.write(f"{name}\t{fconf}\t{v_frame:.3f}\t{fmax:.4f}\t{best}\t{frame_f2[best]:.4f}\t{v_best:.3f}\t{v_best - v_frame:+.3f}\t{verdict} -> conf {chosen}\n")
        print(f"[{name:24s}] frame op {fconf} (vF2 {v_frame:.3f}, frame F2 {frame_f2[fconf]:.3f}/max {fmax:.3f}) | best eligible alarm op {best} (vF2 {v_best:.3f}) gain {v_best - v_frame:+.3f} -> {verdict} (conf {chosen})", flush=True)
    out.close(); print("->", OUT_TSV)


if __name__ == "__main__":
    main()
