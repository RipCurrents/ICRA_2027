#!/usr/bin/env python3
"""Export student checkpoints to ONNX at the resolution ladder used on the edge device.

For every checkpoint matching CKPT_GLOB (local best_models/ + cluster_pull/best_models/),
exports ONNX (opset 17, static batch 1, onnxslim, FP32) at each IMGSZ, records params /
GFLOPs / file size in results/export_manifest.tsv. TensorRT FP16/INT8 engines are built ON
THE DEVICE from these ONNX files (trtexec), never here. Runs on CPU so it never competes
with local training. Env: conda yoloV26 (ultralytics 8.4.92, onnx, onnxslim). No CLI args."""
import glob
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIRS = [os.path.join(HERE, "best_models"), "/mnt/linux/icra_edge/cluster_pull/best_models"]
CKPT_GLOB = "*.pt"
ONLY = None  # substring filter on stems, e.g. "canon_bfi_n_s1"; None = all
IMGSZ = (640, 512, 384)
OUT_DIR = "/mnt/linux/icra_edge/exports"  # onnx files (bulky)
MANIFEST = os.path.join(HERE, "results", "export_manifest.tsv")


def main():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.isfile(MANIFEST):
        done = {tuple(l.split("\t")[1:3]) for l in open(MANIFEST).read().splitlines()[1:]}
    else:
        open(MANIFEST, "w").write("timestamp\tstem\timgsz\tparams_M\tgflops\tonnx_MB\tonnx_path\n")
    ckpts = sorted(c for d in CKPT_DIRS if os.path.isdir(d) for c in glob.glob(os.path.join(d, CKPT_GLOB)))
    for ck in ckpts:
        stem = os.path.splitext(os.path.basename(ck))[0]
        if ONLY and ONLY not in stem:
            continue
        model = None
        for sz in IMGSZ:
            if (stem, str(sz)) in done:
                continue
            model = model or YOLO(ck)
            out = model.export(format="onnx", imgsz=sz, opset=17, simplify=True, dynamic=False,
                               half=False, batch=1, device="cpu")
            dst = os.path.join(OUT_DIR, f"{stem}_{sz}.onnx")
            os.replace(out, dst)
            params = get_num_params(model.model) / 1e6
            flops = get_flops(model.model, sz)
            with open(MANIFEST, "a") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{stem}\t{sz}\t{params:.3f}\t{flops:.2f}\t"
                        f"{os.path.getsize(dst) / 1e6:.1f}\t{dst}\n")
            print(f"{stem} @{sz}: {params:.2f}M params, {flops:.1f} GFLOPs, {os.path.getsize(dst) / 1e6:.1f} MB")
    print(f"-> {MANIFEST}")


if __name__ == "__main__":
    main()
