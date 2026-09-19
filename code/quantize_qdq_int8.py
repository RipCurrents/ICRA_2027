#!/usr/bin/env python3
"""Explicit Q/DQ INT8 ONNX exports for TensorRT 10.3 (its implicit calibrator is broken - TRT 10.3
checkSanity assertion, fixed in 10.4; device NOTES 2026-09-06). onnxruntime static QDQ PTQ (symmetric int8, per-channel weights, Conv/MatMul
only - the placement TensorRT explicit quantization accepts; ModelOpt hung) with entropy calibration on the FROZEN 500-TRAIN-frame set (device_kit/calib_640, letterboxed with the kit's
board_bench.letterbox at each input size). Output: <stem>_qdq_int8.onnx per graph in OUT_DIR + a local
onnxruntime sanity (Q/DQ int8 vs fp32 on N_CHECK real frames: score/box agreement). Constants at top,
no CLI args. Env: rfdetr (+ nvidia-modelopt[onnx])."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KIT = "/mnt/linux/icra_edge/device_kit"
CALIB_DIR = os.path.join(KIT, "calib_640")
CALIB_LIST = os.path.join(HERE, "..", "..", "ICRA_2027", "device_kit_updates", "int8_calib_strided128.txt")  # STRIDED 128 of the
                             # frozen train500 (both classes; the FIRST 128 were all no-rips - device intake catch 2026-09-06)
MODELS = [f"{m}_{sz}" for m in ("t1_nano_k025_t024_s1_n_s1_best_e2", "canon_bfi_n_s1_best_e18") for sz in (384, 512, 640)]
OUT_DIR = os.path.join(KIT, "models_qdq_int8")
N_CALIB = 128                 # strided 128 of the frozen train500 (ModelOpt on 500 hung at the static-quant step 2026-09-06)
CALIB_METHOD = "entropy"      # onnxruntime CalibrationMethod.Entropy
N_CHECK = 20


def calib_batch(paths, sz):
    import cv2
    from board_bench import letterbox
    xs = [letterbox(cv2.imread(p), sz)[0] for p in paths]
    return np.concatenate(xs, 0).astype(np.float32)


def main():
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, paths, sz, input_name):
            self.paths, self.sz, self.name, self.i = paths, sz, input_name, 0
        def get_next(self):
            if self.i >= len(self.paths):
                return None
            x = calib_batch([self.paths[self.i]], self.sz); self.i += 1
            return {self.name: x}
        def rewind(self):
            self.i = 0
    os.makedirs(OUT_DIR, exist_ok=True)
    rel = [l.strip() for l in open(CALIB_LIST) if l.strip()][:N_CALIB]
    calib_paths = [os.path.join(CALIB_DIR, r) for r in rel]
    test_imgs = json.load(open(f"{KIT}/test.json"))["images"][:N_CHECK]
    check_paths = [f"{KIT}/test_images_640/" + im["file_name"].lstrip("./") for im in test_imgs]
    for stem in MODELS:
        sz = int(stem.rsplit("_", 1)[1])
        src, dst = f"{KIT}/models/{stem}.onnx", f"{OUT_DIR}/{stem}_qdq_int8.onnx"
        if os.path.isfile(dst):
            print(f"[{stem}] exists"); continue
        print(f"[{stem}] calibrating on {len(calib_paths)} train frames at {sz}", flush=True)
        pre = dst.replace("_qdq_int8.onnx", "_pre.onnx")
        quant_pre_process(src, pre, skip_symbolic_shape=False)
        iname = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).get_inputs()[0].name
        quantize_static(pre, dst, Reader(calib_paths, sz, iname), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
                        reduce_range=False, op_types_to_quantize=["Conv", "MatMul"],
                        calibrate_method=CalibrationMethod.Entropy,
                        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                                       "CalibMovingAverage": False,
                                       "QuantizeBias": False})   # TRT 10.3 rejects INT32-bias DQ nodes
                                                                 # (device NOTES 2026-09-06 15:23); biases stay fp32
        os.remove(pre)
        s0 = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
        s1 = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
        n = s0.get_inputs()[0].name
        agree, tot = 0, 0
        for x in (calib_batch([p], sz) for p in check_paths):
            o0, o1 = s0.run(None, {n: x})[0][0], s1.run(None, {n: x})[0][0]
            k0, k1 = (o0[:, 4] >= 0.04).sum(), (o1[:, 4] >= 0.04).sum()
            agree += int(k0 == k1); tot += 1
        print(f"[{stem}] -> {dst}; det-count agreement at conf .04 on {tot} frames: {agree}/{tot}", flush=True)
    print("QDQ INT8 EXPORTS DONE ->", OUT_DIR)


if __name__ == "__main__":
    main()
