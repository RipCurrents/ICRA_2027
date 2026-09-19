#!/usr/bin/env python3
"""Build the T1 ablation views by re-running build_teacher_view.py with different
constants (one after another; each view is loader-verified). Constants-at-top.
Env: conda yoloV26. Skips views whose build_info.json already exists."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("btv", os.path.join(HERE, "build_teacher_view.py"))
btv = importlib.util.module_from_spec(spec)
sys.modules["btv"] = btv
spec.loader.exec_module(btv)

BASE = dict(TEACHER_TAG="rfdetr_seg_nano", TAU=0.24, K_SECONDS=0.25, STRIDE=1,
            KEEP_EMPTY_RIP_FRAMES=False, BOX_SOURCE="rect", INCLUDE_NEGATIVE_VIDEOS=True)
# each entry = overrides on BASE; view name auto-derived by build_teacher_view.auto_name()
CONFIGS = [
    dict(K_SECONDS=0.0),          # no temporal smoothing (single-frame teacher labels)
    dict(STRIDE=2),               # density curve
    dict(STRIDE=4),
    dict(TAU=0.15),               # threshold sensitivity (recall-leaning)
    dict(TAU=0.35),               # (precision-leaning)
    dict(KEEP_EMPTY_RIP_FRAMES=True),   # teacher misses become negatives
    dict(TEACHER_TAG="rfdetr_seg_nano_add"),  # +additional-data teacher (needs its npz pass)
]


def main():
    for cfg in CONFIGS:
        for k, v in {**BASE, **cfg}.items():
            setattr(btv, k, v)
        btv.PRED_DIR = os.path.join("/mnt/linux/icra_edge/predictions", "teacher_" + btv.TEACHER_TAG)
        btv.VIEW_NAME = None
        name = btv.auto_name()
        if os.path.isfile(os.path.join(btv.VIEWS_ROOT, name, "build_info.json")):
            print(f"SKIP {name}: exists", flush=True)
            continue
        if not os.path.isdir(btv.PRED_DIR) or len(os.listdir(btv.PRED_DIR)) < 165:
            print(f"SKIP {name}: teacher predictions incomplete in {btv.PRED_DIR}", flush=True)
            continue
        print(f"=== building {name} ({cfg})", flush=True)
        btv.main()
    print("ABLATION_VIEWS_DONE", flush=True)


if __name__ == "__main__":
    main()
