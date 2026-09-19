#!/usr/bin/env python3
"""Teacher-free views for the semi-supervised HOLDOUT (design: ICRA_2027/notes/design_semisup_holdout.md).

Filters existing, loader-verified views to the LABELLED half of manifests/holdout_h50_seed0.tsv:
  h50_sparse  <- canonical_bfi (expert labels on the sampled frames; labelled videos only, rip and no-rip)
  h50_interp  <- interp_gt_s1 (interpolated expert masks -> boxes on every native frame + the expert frames +
                 dense negatives; labelled videos only)
Nothing from an UNLABELLED video enters either view (the unlabelled no-rip videos are unlabelled too: only the
teacher view may use them). Images/labels are symlinked to the source views' trees (label files are per image
and unaffected by the filter); train.txt is the filtered list; val/test manifests are copied unchanged; the
ultralytics scan .cache lands inside the new view (labels/ dirs are real down to the symlinked leaves).
Loader-verified through ultralytics' YOLODataset. Constants below, no CLI args. Env: conda yoloV26.
"""
import csv
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS_ROOT = "/home/user/RipBench/icra_edge_views"
SPLIT_TSV = os.path.join(HERE, "manifests", "holdout_h50_seed0.tsv")
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
GT_LABELS = os.path.join(DATASET_ROOT, "labels", "bbox_from_instance", "yolo26", "labels", "sampled_frames")
SOURCES = {"h50_sparse": "canonical_bfi", "h50_interp": "interp_gt_s1", "h50_interp_negall": "interp_gt_s1"}
NEG_FROM_UNLABELLED_NORIP = {"h50_interp_negall"}   # control (b+): also keep the UNLABELLED no-rip videos' dense negative
                                                    # frames (their VIDEO-LEVEL label only, no teacher) - critic memo 2026-08-27
VERIFY = True
VIDEO_RE = re.compile(r"(RipBench-(?:NR-)?\d+)_\d+\.jpg$")


def build(name, src_name, labelled, extra_neg=frozenset()):
    src = os.path.join(VIEWS_ROOT, src_name)
    view = os.path.join(VIEWS_ROOT, name)
    if os.path.isdir(view):
        shutil.rmtree(view)
    os.makedirs(os.path.join(view, "images"))
    os.makedirs(os.path.join(view, "labels", "sampled_frames"))
    # images: mirror the source view's images/ entries (sampled_frames -> dataset, dense_frames -> extracted frames)
    src_images = os.path.join(src, "images")
    if os.path.islink(src_images):                         # canonical: images -> dataset images dir
        os.rmdir(os.path.join(view, "images"))
        os.symlink(os.readlink(src_images), os.path.join(view, "images"))
    else:
        for e in os.listdir(src_images):
            os.symlink(os.path.realpath(os.path.join(src_images, e)), os.path.join(view, "images", e))
    for cls in ("rips", "no-rips"):
        os.symlink(os.path.join(GT_LABELS, cls), os.path.join(view, "labels", "sampled_frames", cls))
    # labels/dense_frames: a REAL dir holding only the kept videos' label files (hard links, same NVMe) so the
    # cluster tar (ship_to_cluster.sh) carries exactly the filtered set; h50_sparse gets an empty dir + gt_train.txt
    # so it ships and rebuilds on the node like the dense views
    os.makedirs(os.path.join(view, "labels", "dense_frames"))
    src_dense = os.path.realpath(os.path.join(src, "labels", "dense_frames")) if os.path.isdir(os.path.join(src, "labels", "dense_frames")) else None
    kept = dropped = linked = 0
    with open(os.path.join(view, "train.txt"), "w") as out:
        for line in open(os.path.join(src, "train.txt")):
            line = line.strip()
            if not line:
                continue
            v = VIDEO_RE.search(line).group(1)
            if v in labelled or v in extra_neg:
                # absolute paths (the cluster rebuild remaps '<view_root>/images/' -> node view; canonical_bfi's list is './images/...')
                out.write((view + line[1:] if line.startswith("./") else line.replace(src, view)) + "\n")
                kept += 1
                if src_dense and "/images/dense_frames/" in line:
                    lab = os.path.join(src_dense, os.path.basename(line)[:-4] + ".txt")
                    if os.path.isfile(lab):
                        os.link(lab, os.path.join(view, "labels", "dense_frames", os.path.basename(lab)))
                        linked += 1
            else:
                dropped += 1
    gt_src = os.path.join(src, "gt_train.txt") if os.path.isfile(os.path.join(src, "gt_train.txt")) else os.path.join(src, "train.txt")
    with open(os.path.join(view, "gt_train.txt"), "w") as out:
        for line in open(gt_src):
            if line.strip() and VIDEO_RE.search(line).group(1) in labelled:
                l = line.strip()
                out.write((view + l[1:] if l.startswith("./") else l.replace(src, view)) + "\n")
    print(f"  {linked} dense label files hard-linked", flush=True)
    for split in ("val", "test"):
        with open(os.path.join(view, f"{split}.txt"), "w") as out:
            out.write(open(os.path.join(src, f"{split}.txt")).read().replace(src, view))
    with open(os.path.join(view, "data.yaml"), "w") as f:
        f.write(f"path: {view}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: rip_current\n")
    # count label boxes of the kept images
    n_boxes = n_lab = 0
    for line in open(os.path.join(view, "train.txt")):
        p = line.strip()
        p = p if p.startswith("/") else os.path.join(view, p[2:] if p.startswith("./") else p)
        lp = p.replace("/images/", "/labels/", 1)[:-4] + ".txt"
        if os.path.isfile(lp):
            n = sum(1 for l in open(lp) if l.strip())
            n_boxes += n
            n_lab += 1
    info = dict(view=view, source=src, split=SPLIT_TSV, n_train_images=kept, n_dropped_unlabelled=dropped,
                n_label_files=n_lab, n_boxes=n_boxes, built=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(info, open(os.path.join(view, "build_info.json"), "w"), indent=1)
    print(f"[{name}] from {src_name}: kept {kept} train images ({dropped} dropped as unlabelled), {n_lab} label files, {n_boxes} boxes", flush=True)
    return view, kept, n_boxes


def verify(view, expected_imgs, expected_boxes):
    from ultralytics.data.dataset import YOLODataset
    ds = YOLODataset(img_path=os.path.join(view, "train.txt"), data={"names": {0: "rip_current"}}, task="detect", imgsz=640, augment=False)
    n_img = len(ds.labels)
    n_box = sum(len(l["cls"]) for l in ds.labels)
    print(f"  loader: {n_img} images, {n_box} boxes (expected {expected_imgs} / {expected_boxes})", flush=True)
    assert n_img == expected_imgs and n_box == expected_boxes, "LOADER MISMATCH"


def main():
    rows = list(csv.DictReader(open(SPLIT_TSV), delimiter="\t"))
    labelled = {r["video"] for r in rows if r["role"] == "labelled"}
    print(f"{len(labelled)} labelled videos of {len(rows)}")
    for name, src in SOURCES.items():
        unl_norip = {r["video"] for r in rows if r["role"] == "unlabelled" and r["kind"] == "no-rip"}
        view, n_img, n_box = build(name, src, labelled, unl_norip if name in NEG_FROM_UNLABELLED_NORIP else frozenset())
        if VERIFY:
            verify(view, n_img, n_box)


if __name__ == "__main__":
    main()
