#!/usr/bin/env python3
"""Condition (c) of the semi-supervised holdout: the T1 teacher view built with the HOLDOUT teacher
(RF-DETR-Seg-nano retrained on the labelled half; design: ICRA_2027/notes/design_semisup_holdout.md).

Drives build_teacher_view.py (imported, constants overridden here, nothing else changed):
  * teacher tag rfdetr_seg_nano_h50 -> candidates from teacher_predict_frames.py run with
    EDGE_TEACHER_CKPT=<h50 teacher> EDGE_TEACHER_TAG=rfdetr_seg_nano_h50 over ALL 165 train videos;
  * GT_ONLY_VIDEOS = the 83 LABELLED videos: only their expert frames enter the view;
  * TEACHER_ONLY_VIDEOS = the 82 UNLABELLED videos: their VIDEO-LEVEL label (rips/ vs no-rips/) is NOT used -
    every frame goes through the teacher; confident boxes -> positives, all other frames -> negatives (kept).
    (Before the critic memo of 2026-08-27 the builder would have made the unlabelled no-rip videos dense
    negatives by folder name alone, which is not a "labels missing" scenario.) The labelled half is the plain
    T1 recipe (expert frames + teacher on the rest; its no-rip videos as dense negatives by their known label);
  * TAU = the h50 teacher's val-selected F2 operating point on the bfi view (results/teacher_h50_boxified_eval.tsv,
    produced by teacher_h50_eval.py) - set it below before building;
  * same K_SECONDS 0.25 / NMS 0.5 / rect boxes / drop-empty-rip-frames as the paper's dense view.
View name h50_teacher; loader-verified by the builder. Constants below, no CLI args. Env: conda yoloV26.
"""
import csv
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEACHER_TAG = "rfdetr_seg_nano_h50"
TAU = 0.23   # from results/teacher_h50_boxified_eval.tsv (val row, selected_conf of the epoch-6 winner; set 2026-08-28 06:3x)
VIEW_NAME = "h50_teacher"
DROP_EMPTY_UNLABELLED = False   # True -> view h50_teacher_dropempty: teacher-empty frames of the unlabelled videos are DROPPED (critic discriminator)
SPLIT_TSV = os.path.join(HERE, "manifests", "holdout_h50_seed0.tsv")


def main():
    assert TAU is not None, "set TAU to the h50 teacher's val-selected conf first (teacher_h50_eval.py)"
    spec = importlib.util.spec_from_file_location("btv_h50", os.path.join(HERE, "build_teacher_view.py"))
    btv = importlib.util.module_from_spec(spec)
    sys.modules["btv_h50"] = btv
    spec.loader.exec_module(btv)
    labelled = {r["video"] for r in csv.DictReader(open(SPLIT_TSV), delimiter="\t") if r["role"] == "labelled"}
    assert len(labelled) == 83, len(labelled)
    btv.TEACHER_TAG = TEACHER_TAG
    btv.PRED_DIR = os.path.join("/mnt/linux/icra_edge/predictions", "teacher_" + TEACHER_TAG)
    assert os.path.isdir(btv.PRED_DIR) and len([f for f in os.listdir(btv.PRED_DIR) if f.endswith(".npz")]) == 165, btv.PRED_DIR
    btv.TAU = TAU
    btv.GT_ONLY_VIDEOS = labelled
    unlabelled = {r["video"] for r in csv.DictReader(open(SPLIT_TSV), delimiter="\t") if r["role"] == "unlabelled"}
    assert len(unlabelled) == 82, len(unlabelled)
    btv.TEACHER_ONLY_VIDEOS = unlabelled
    btv.TEACHER_ONLY_KEEP_EMPTY = not DROP_EMPTY_UNLABELLED
    if DROP_EMPTY_UNLABELLED:
        btv.VIEW_NAME = VIEW_NAME + "_dropempty"
    btv.VIEW_NAME = VIEW_NAME if not DROP_EMPTY_UNLABELLED else VIEW_NAME + "_dropempty"
    print(f"building {VIEW_NAME}: teacher {TEACHER_TAG}, tau {TAU}, expert frames from {len(labelled)} labelled videos; {len(unlabelled)} videos teacher-only (no video-level label)", flush=True)
    btv.main()


if __name__ == "__main__":
    main()
