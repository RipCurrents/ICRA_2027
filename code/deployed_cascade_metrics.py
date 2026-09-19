#!/usr/bin/env python3
"""Critic W1/Q1: alarm-level metrics of the DECLARED deployed configuration from the board's own outputs -
gate@224 thr 0.50 through the deployed preprocessing (results/gate_deployed_path_test_probs.json, x86 = board code
path) + the Orin FP16 t1@384 test dump at conf 0.04, replayed at 1 Hz (k = n = 1, GAP_S episode merge), against the
sparse-label baseline (canonical@384, same board/dump protocol) at equal duty, plus the ungated and 0.2 Hz variants.
Video-level bootstrap (B=1000, seed 0) over rip videos (recall, latency) and no-rip clips (false-episode fraction).
Writes results/deployed_cascade_test.tsv + the predeclared-gate verdict table results/deployment_gates.tsv.
Constants at top, no CLI args. Env: yoloV26."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
GT_TEST = os.path.join(DATASET_ROOT, "labels", "bbox_from_instance", "coco", "test.json")
GATE_PREPROC = "centercrop"   # v2 (declared 2026-09-07 pending the author) | "letterbox" = v1 rows already reported
GATE_PROBS = os.path.join(HERE, "results", f"gate_deployed_path_test_probs_{GATE_PREPROC}.json")
DR = os.path.join(REPO, "ICRA_2027", "device_results")
ASSEMBLED = "/mnt/linux/icra_edge/device_dumps_assembled"   # split .gz parts from device-week concatenated (gzip -t verified)
# v2 (2026-09-12): TIGHT-view frame confs for every row (expert yolo26s .07, dense yolo26s .02 - v1 wrongly used their BFI confs
# .05/.01, caught by Codex's home-takeover audit); RF-DETR rows from the INTER_AREA re-dumps (REPLY 23); extended ladder rows added.
DUMPS = {"deployed t1@384 (Orin FP16)": (os.path.join(DR, "t1_384_fp16_orin_preds_test.json"), 0.04),
         "sparse-label baseline canon@384 (Orin FP16)": (os.path.join(DR, "canon_bfi_384_fp16_orin_preds_test.json"), 0.03),
         "board-fitted Orin: dense yolo26s@384 (FP16)": (os.path.join(DR, "s_s3_384_fp16_orin_preds_test.json"), 0.02),
         "board-fitted Orin: expert yolo26s@384 (FP16)": (os.path.join(DR, "extended_20260907", "orin", "measurements", "canon_bfi_s_s1_best_e1_384_orin_fp16_preds_test.json"), 0.07),
         "board-fitted Orin: RF-DETR-n@384 (FP16, val winner, INTER_AREA)": (os.path.join(DR, "rf_area_20260908", "orin", "measurements", "rfdetr_n_det_640_orin_fp16_preds_test.json.gz"), 0.23),
         "board-fitted Orin: RF-DETR-s@512 (FP16, INTER_AREA)": (os.path.join(DR, "rf_area_20260908", "orin", "measurements", "rfdetr_s_det_512_orin_fp16_preds_test.json.gz"), 0.26),
         "board-fitted Orin: RF-DETR-m@576 (FP16, INTER_AREA)": (os.path.join(DR, "rf_area_20260908", "orin", "measurements", "rfdetr_m_det_576_orin_fp16_preds_test.json.gz"), 0.31),
         "board-fitted RPi 5: RF-DETR-n@384 (ORT CPU, val winner, INTER_AREA, 400 ms)": (os.path.join(ASSEMBLED, "rfdetr_n_det_640_rpi5_fan_area_preds_test.json.gz"), 0.23),
         "board-fitted RPi 5: RF-DETR-s@512 (ORT CPU, INTER_AREA, 818 ms)": (os.path.join(ASSEMBLED, "rfdetr_s_det_512_rpi5_fan_area_preds_test.json.gz"), 0.26),
         "board-fitted RPi 4 heatsink: expert yolo26s@384 (ORT CPU, val winner among fitting, 411 ms)": (os.path.join(DR, "pi4_parallel_20260908", "heatsink", "cpu_dumps", "canon_bfi_s_s1_best_e1_384_rpi4_heatsink_preds_test.json"), 0.07),
         "board-fitted RPi 4 bare: expert yolo26s@384 (ORT CPU, 424 ms)": (os.path.join(DR, "pi4_parallel_20260908", "bare", "cpu_dumps", "canon_bfi_s_s1_best_e1_384_rpi4_bare_preds_test.json"), 0.07),
         "board-fitted Jetson Nano: expert yolo26s@384 (TRT 8.2 FP16, 53 ms)": (os.path.join(DR, "nano_extended_20260907", "dumps", "canon_bfi_s_s1_best_e1_384_trt82_nano_fp16_preds_test.json.gz"), 0.07),
         "i.MX95 CPU fallback: dense yolo26n@384 (ORT FP32)": (os.path.join(DR, "imx95_deployment_20260908", "imx95", "cpu_fallback", "t1_nano_k025_t024_s1_n_s1_best_e2_384_imx95_cpu_fp32_preds_test.json"), 0.04),
         "i.MX95 NPU: QAT2-adapted dense yolo26n@384 (native images)": (os.path.join(DR, "nxp_selected_test_20260910", "qat2_preds_test.json.gz"), 0.04),
         }
CONFIGS = [("gate .50, 1 Hz", 0.50, 1.0), ("no gate, 1 Hz", None, 1.0), ("gate .50, 0.2 Hz", 0.50, 0.2), ("gate .50, 5 Hz", 0.50, 5.0)]
K, N = 1, 1
B, SEED = 1000, 0
OUT_TSV = os.path.join(HERE, "results", f"deployed_cascade_test_{GATE_PREPROC}.tsv")
GATES_TSV = os.path.join(HERE, "results", f"deployment_gates_{GATE_PREPROC}.tsv")


def main():
    import event_metrics as em
    from pycocotools.coco import COCO
    gt = COCO(GT_TEST)
    frames, first_gt, fps, _ = em.prep(gt, "test")
    by_name = {im["id"]: os.path.basename(im["file_name"]) for im in gt.dataset["images"]}
    probs = json.load(open(GATE_PROBS))
    cls = {iid: probs[b] for iid, b in by_name.items()}
    rng = np.random.default_rng(SEED)
    rip = [v for v in frames if "NR-" not in v]; nr = [v for v in frames if "NR-" in v]
    out = open(OUT_TSV, "w")
    out.write("model\tconfig\trip_alarmed\tci_lo\tci_hi\tlatency_median_s\tlatency_p90_s\tnorip_false_episode\tci_lo\tci_hi\tdetector_runs_frac\tvideo_F2\n")
    res = {}
    for mname, (dump, conf) in DUMPS.items():
        import gzip
        pt = json.load((gzip.open if dump.endswith('.gz') else open)(dump, 'rt'))
        positives = {p["image_id"]: True for p in pt if p["score"] >= conf}
        for cname, gate, duty in CONFIGS:
            m = em.evaluate(frames, first_gt, fps, cls, positives, gate, duty, K, N)
            # per-video outcomes for the bootstrap
            gate_open = {} if gate is None else {iid: (pr >= gate) for iid, pr in cls.items()}
            hit = {}
            for v in rip:
                alarms, _ = em.replay(frames[v], positives, fps[v], duty, gate_open, K, N)
                t0 = first_gt.get(v, frames[v][0][0]) / fps[v]
                hit[v] = any(a >= t0 for a in alarms)
            fa = {}
            for v in nr:
                alarms, _ = em.replay(frames[v], positives, fps[v], duty, gate_open, K, N)
                fa[v] = len(alarms) > 0
            rec_bs = [np.mean([hit[v] for v in rng.choice(rip, len(rip))]) for _ in range(B)]
            fa_bs = [np.mean([fa[v] for v in rng.choice(nr, len(nr))]) for _ in range(B)]
            r = dict(rip=float(m["rip_alarmed_frac"]), rip_lo=np.percentile(rec_bs, 2.5), rip_hi=np.percentile(rec_bs, 97.5),
                     lat_med=m["latency_median_s"], lat_p90=m["latency_p90_s"],
                     fa=float(m["norip_alarmed_frac"]), fa_lo=np.percentile(fa_bs, 2.5), fa_hi=np.percentile(fa_bs, 97.5),
                     det=m["detector_runs_frac"], vf2=m["video_F2"])
            res[(mname, cname)] = r
            out.write(f"{mname}\t{cname}\t{r['rip']:.3f}\t{r['rip_lo']:.3f}\t{r['rip_hi']:.3f}\t{r['lat_med']}\t{r['lat_p90']}\t{r['fa']:.3f}\t{r['fa_lo']:.3f}\t{r['fa_hi']:.3f}\t{r['det']}\t{r['vf2']}\n")
            print(f"[{mname[:28]:28s} | {cname:16s}] rip alarmed {r['rip']:.3f} [{r['rip_lo']:.3f},{r['rip_hi']:.3f}]  TTA med/p90 {r['lat_med']}/{r['lat_p90']} s  no-rip false-episode {r['fa']:.3f} [{r['fa_lo']:.3f},{r['fa_hi']:.3f}]  detector runs {r['det']}", flush=True)
    out.close()
    d = res[("deployed t1@384 (Orin FP16)", "gate .50, 1 Hz")]; b = res[("sparse-label baseline canon@384 (Orin FP16)", "gate .50, 1 Hz")]
    gates = [
        ("G1 rip-video recall within 0.05 of the sparse-label baseline at equal duty", f"{d['rip']:.3f} vs {b['rip']:.3f}", "PASS" if d['rip'] >= b['rip'] - 0.05 else "FAIL"),
        ("G2 false-alarm no-rip clips <= 0.15 with the gate", f"{d['fa']:.3f} [{d['fa_lo']:.3f},{d['fa_hi']:.3f}]", "PASS" if d['fa'] <= 0.15 else "FAIL"),
        ("G3 median time-to-alarm <= 5 s and p90 <= 15 s at 1 Hz", f"{d['lat_med']} / {d['lat_p90']} s", "PASS" if d['lat_med'] and float(d['lat_med']) <= 5 and float(d['lat_p90']) <= 15 else "FAIL"),
        ("G4 <= 1 s per decision on the board, accuracy within 0.005 F2 of the workstation", "6.1 ms cached pipeline (pre 3.16 + infer 2.84 + post 0.13, MAXN in-process; engine alone 2.44 ms); F2 +0.006/+0.021 vs PyTorch ref (favourable), <=0.0002 vs deployed-graph ref", "PASS latency; accuracy FAIL upward vs declared reference"),
        ("G5 a 20 Wh battery-day at 0.2 decisions/s", "Orin 4.98 W gross -> ~4 h", "FAIL (platform idle)"),
    ]
    with open(GATES_TSV, "w") as f:
        f.write("gate\tmeasured\tverdict\n")
        for g in gates: f.write("\t".join(g) + "\n")
    for g in gates: print("  ", g)
    print("wrote", OUT_TSV, "and", GATES_TSV)


if __name__ == "__main__":
    main()
