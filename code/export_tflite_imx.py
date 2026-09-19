#!/usr/bin/env python3
"""int8 TFLite export for the i.MX95 (eIQ/Neutron takes int8 TFLite, not ONNX). Exports the dense yolo26n student
(best seed by val), the canonical yolo26n reference and the cls gate; ultralytics format='tflite' int8 with a
calibration subset; then validates each artefact with the tflite CPU interpreter on VAL_N val frames vs the PyTorch dump
(box-level sanity + F2 at the frozen conf on that subset - a SANITY check, not a benchmark row; the benchmark row comes
from the board). Constants below, no CLI args. Env: icra_tflite (see setup block in the launcher log)."""
import os
import shutil
MODELS = {  # name -> checkpoint (from /mnt/linux/icra_edge/best_models, same files the ONNX export used)
    "t1_n_s1": "/mnt/linux/icra_edge/best_models/t1_nano_k025_t024_s1_n_s1_best_e2.pt",
    "canon_n_s1": "/mnt/linux/icra_edge/best_models/canon_bfi_n_s1_best_e18.pt",
    "gate_cls224": "/home/user/RipBench/RipBench_v1.2.0/models/classification/best_models/3_cls_yolo26_nano_224_best_e50_j2856380.pt",  # the cascade gate = benchmark cls nano 224 (the .onnx in exports/ came from this)
}
OUT_DIR = "/mnt/linux/icra_edge/exports/tflite_imx"
IMG_SZ = {"t1_n_s1": 640, "canon_n_s1": 640, "gate_cls224": 224}
CALIB_YAML = None      # int8 calibration uses each model's own data yaml fraction if None (ultralytics default)
VAL_N = 200            # local sanity frames
os.makedirs(OUT_DIR, exist_ok=True)
from ultralytics import YOLO
for name, ckpt in MODELS.items():
    m = YOLO(ckpt)
    path = m.export(format="tflite", int8=True, imgsz=IMG_SZ[name], data=CALIB_YAML) if CALIB_YAML else \
           m.export(format="tflite", int8=True, imgsz=IMG_SZ[name])
    dst = os.path.join(OUT_DIR, f"{name}_int8.tflite")
    shutil.move(str(path), dst) if os.path.isfile(str(path)) else print(f"[{name}] export returned {path}")  # shutil.move: dst is on another filesystem (os.replace fails cross-device)
    print(f"[{name}] -> {dst}", flush=True)
print("EXPORTS DONE - validation follows in validate_tflite_imx.py")
