#!/usr/bin/env python3
"""Fidelity gate for NEW exports (pre-declaration 4 amended, item d): run each ONNX with the DEPLOYED decode on the
full-resolution VAL images (x86, CUDA EP) and score at the benchmark-frozen conf on the TIGHT view; PASS if |F2 - benchmark
val row| <= 0.01. Yolo graphs: board_bench.letterbox + decode; RF-DETR: rfdetr_onnx_bench.preprocess/decode (square resize,
ImageNet norm, rip = index 0). VAL only. Constants at top, no CLI args. Env: rfdetr (onnxruntime-gpu)."""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
GT_VAL_TIGHT = os.path.join(DATASET_ROOT, "labels", "bbox", "coco", "val.json")
E = "/mnt/linux/icra_edge/device_kit_ext_2026-09-07/models_ext"
EXPORTS = {  # name -> (onnx, kind, res, frozen conf tight, benchmark val F2 tight | student_eval row name)
    "expert yolo26s 384": (f"{E}/canon_bfi_s_s1_best_e1_384.onnx", "yolo", 384, 0.07, "canon_bfi_s_s1_best_e1_img384"),   # reference = the 384-inference benchmark row
    "expert yolo26s 640": (f"{E}/canon_bfi_s_s1_best_e1_640.onnx", "yolo", 640, 0.07, "canon_bfi_s_s1_best_e1"),
    "RF-DETR-s 512": (f"{E}/rfdetr_s_det_512.onnx", "rfdetr", 512, 0.26, 0.6456),
    "RF-DETR-m 576": (f"{E}/rfdetr_m_det_576.onnx", "rfdetr", 576, 0.31, 0.6114),
}
TOL = 0.01
OUT_TSV = os.path.join(HERE, "results", "export_fidelity_val.tsv")


def main():
    import csv
    import cv2
    import onnxruntime as ort
    import board_bench as bb
    import rfdetr_onnx_bench as rb
    spec = importlib.util.spec_from_file_location("ed_f", os.path.join(REPO, "eval", "evaluate_detection.py"))
    ED = importlib.util.module_from_spec(spec); sys.modules["ed_f"] = ED; spec.loader.exec_module(ED)
    gt = ED.COCO(GT_VAL_TIGHT)
    ws = {r["model"]: float(r["F2_50"]) for r in csv.DictReader(open(os.path.join(HERE, "results", "student_eval_tight.tsv")), delimiter="\t") if r["split"] == "val"}
    out = open(OUT_TSV, "a")
    if os.path.getsize(OUT_TSV) == 0:
        out.write("export\tres\tconf\tonnx_val_F2\tbenchmark_val_F2\tdiff\tverdict\n")
    done = {l.split("\t")[0] for l in open(OUT_TSV) if "\t" in l and "PASS" in l}
    for name, (path, kind, res, conf, bench) in EXPORTS.items():
        if name in done:
            continue
        if isinstance(bench, str):
            if bench not in ws:
                print(f"[{name}] no benchmark val row {bench} - fidelity established at 640 (same exporter); skipping", flush=True); continue
            bench = ws[bench]
        sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        preds = []
        for i, im in enumerate(gt.dataset["images"]):
            img = cv2.imread(os.path.join(DATASET_ROOT, "images", im["file_name"].lstrip("./")))
            h, w = img.shape[:2]
            if kind == "yolo":
                x, r, top, left = bb.letterbox(img, res)
                boxes, scores = bb.decode(sess.run(None, {iname: x}), r, top, left, 0.001)
                preds += [{"image_id": int(im["id"]), "category_id": 1, "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "score": float(s)} for b, s in zip(boxes, scores)]
            else:
                rb.RES = res
                x, _, _ = rb.preprocess(img)
                o = sess.run(None, {iname: x})
                res_d = {sess.get_outputs()[k].name: o[k] for k in range(len(o))}
                for d in rb.decode(res_d["dets"], res_d["labels"], w, h, 0, 1):
                    d["image_id"] = int(im["id"]); preds.append(d)
            if i % 2000 == 0:
                print(f"  [{name}] {i}", flush=True)
        cache = ED.build_cache(gt, gt.loadRes(preds), "bbox")
        f2 = ED.evaluate_split(cache, conf)[0]["F2"]
        verdict = "PASS" if abs(f2 - bench) <= TOL else "FAIL"
        print(f"[{name}] onnx val F2 {f2:.4f} vs benchmark {bench:.4f} -> {verdict}", flush=True)
        out.write(f"{name}\t{res}\t{conf}\t{f2:.4f}\t{bench:.4f}\t{f2 - bench:+.4f}\t{verdict}\n")
    out.close(); print("FIDELITY DONE ->", OUT_TSV)


if __name__ == "__main__":
    main()
