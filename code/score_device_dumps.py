#!/usr/bin/env python3
"""Score the device sessions' full-test prediction dumps (ICRA_2027/device_results/*_preds_test.json)
with the benchmark evaluator at the VAL-FROZEN confs from the 4090 rows (operating_points.tsv), both
GT views -> results/device_eval_{bfi,tight}.tsv. Test-only path (USE_VAL_SELECTION=False). Skips
dumps already scored. Constants at top, no CLI args. Env: yoloV26."""
import glob
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DUMP_DIR = os.path.join(REPO, "ICRA_2027", "device_results")
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
OUT = {"bbox_from_instance": os.path.join(HERE, "results", "device_eval_bfi.tsv"),
       "bbox": os.path.join(HERE, "results", "device_eval_tight.tsv")}
FROZEN = {  # dump-name prefix -> {view: conf}; ORDER MATTERS (first match wins) - specific prefixes first
    "canon_bfi_s_s1": {"bbox_from_instance": 0.05, "bbox": 0.07},   # expert yolo26s (benchmark val rows)
    "rfdetr_n": {"bbox_from_instance": None, "bbox": 0.23},         # RF-DETR-n: tight view only (benchmark val conf)
    "rfdetr_s": {"bbox_from_instance": None, "bbox": 0.26},         # feasibility only unless re-gated
    "rfdetr_m": {"bbox_from_instance": None, "bbox": 0.31},
    "t1_n_s1": {"bbox_from_instance": 0.04, "bbox": 0.04},
    "t1_": {"bbox_from_instance": 0.04, "bbox": 0.04},        # Orin dumps use the short stem t1_<sz>_<prec>_orin
    "canon_bfi": {"bbox_from_instance": 0.03, "bbox": 0.03},
    "s_s3": {"bbox_from_instance": 0.01, "bbox": 0.02},
    "canon_s_s1": {"bbox_from_instance": None, "bbox": None},   # filled from student_eval TSVs at run time (val-frozen conf of canon_bfi_s_s1)
}

COPY_LONG = 640   # device dumps were made on the 640px-long-side image copies: boxes are in COPY space.
                  # GT is in ORIGINAL space -> rescale analytically (matches build_test_bundle_640.py: r=640/max(w,h),
                  # copy dims = round(dim*r); no-op for images already <= 640).


def rescale_preds(preds, gt_coco):
    import copy as _copy
    dims = {i: (im["width"], im["height"]) for i, im in gt_coco.imgs.items()}
    out = []
    for pr in preds:
        w, h = dims[pr["image_id"]]
        m = max(w, h)
        p2 = _copy.deepcopy(pr)
        if m > COPY_LONG:
            r = COPY_LONG / m
            sx, sy = w / round(w * r), h / round(h * r)
            b = p2["bbox"]
            p2["bbox"] = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        out.append(p2)
    return out


def main():
    spec = importlib.util.spec_from_file_location("ed_dev", os.path.join(REPO, "eval", "evaluate_detection.py"))
    ED = importlib.util.module_from_spec(spec); sys.modules["ed_dev"] = ED; spec.loader.exec_module(ED)
    ED.SAVE_SWEEP, ED.USE_VAL_SELECTION, ED.MODE = False, False, "bbox"
    for task, out_tsv in OUT.items():
        gt_t = ED.COCO(os.path.join(DATASET_ROOT, "labels", task, "coco", "test.json"))
        done = set()
        if os.path.isfile(out_tsv):
            done = {l.split("\t")[1] for l in open(out_tsv) if "\t" in l}
        for p in sorted(glob.glob(os.path.join(DUMP_DIR, "**", "*_preds_test.json"), recursive=True) + glob.glob(os.path.join(DUMP_DIR, "**", "*_preds_test.json.gz"), recursive=True)):
            model = os.path.basename(p).replace("_preds_test.json.gz", "").replace("_preds_test.json", "")
            if model in done:
                continue
            pref = next((k for k in FROZEN if model.startswith(k)), None)
            if pref is None:
                print(f"[{model}] no frozen conf mapping - SKIPPED"); continue
            ED.TASK, ED.OUT_TSV = task, out_tsv
            conf = FROZEN[pref][task]
            if conf is None and pref.startswith("rfdetr"):
                continue       # RF-DETR: no bbox_from_instance benchmark row - tight view only
            if conf is None:   # look up the val-frozen conf of the matching 4090 row
                import csv as _csv
                tsv = os.path.join(HERE, "results", "student_eval_bfi.tsv" if task == "bbox_from_instance" else "student_eval_tight.tsv")
                conf = next(float(r["selected_conf"]) for r in _csv.DictReader(open(tsv), delimiter="\t") if r["model"].startswith("canon_bfi_s_s1_best") and r["split"] == "val")
            ED.FIXED_CONF_THRESHOLD = conf
            import json as _json, tempfile
            import gzip
            opener = gzip.open if p.endswith(".gz") else open
            preds = rescale_preds(_json.load(opener(p, "rt")), gt_t)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                _json.dump(preds, tf); tmp = tf.name
            ED.evaluate_pair(None, gt_t, None, tmp, model)
            os.unlink(tmp)
    print("device dumps scored")

if __name__ == "__main__":
    main()
