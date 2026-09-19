#!/usr/bin/env python3
"""Label-spacing sweep (teacher-free): how sparse can the EXPERT annotation be before interpolation stops working?

For ANCHOR_STRIDES k: keep every k-th annotated (sampled) frame of each rip TRAIN video as an anchor, drop the
others' expert labels, and SDF-interpolate the masks between consecutive anchors with the video-model work's
densifier (build_dense_view.densify, imported from /mnt/linux/RipBench/train/video_models/ytvis_view, unmodified;
gate-validated in that work). The formerly annotated non-anchor frames become interpolated frames like any other
native frame. Outputs one YTVIS-style json per stride:
  /home/user/RipBench/icra_edge_views/spacing/ytvis_dense_interp_a{k}/train.json  (+ anchors_a{k}.txt listing the
  kept expert frames per video). build_interp_view.py then turns each into a yolo view (interp_gt_a{k}) with the
  expert labels restricted to the anchors (GT_ONLY_FRAMES). k=1 reproduces the paper's interp view (sanity check).
Constants below, no CLI args. Env: conda yoloV26 (numpy, pycocotools, cv2). CPU-heavy (SDFs); run as a unit.
"""
import importlib.util
import json
import os
import re
import sys
import time

DENSIFIER = "/mnt/linux/RipBench/train/video_models/ytvis_view/build_dense_view.py"
SPARSE_JSON = "/home/user/RipBench/RipBench_v1.2.0/labels/video_instance_segmentation/ytvis_sparse/train.json"
OUT_ROOT = "/home/user/RipBench/icra_edge_views/spacing"
ANCHOR_STRIDES = (2, 4, 8, 16)
VIDEO_RE = re.compile(r"^(.*)/(RipBench-[\w-]+)_(\d+)\.(\w+)$")


def load_densifier():
    spec = importlib.util.spec_from_file_location("bdv", DENSIFIER)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bdv"] = m
    spec.loader.exec_module(m)
    return m


def subsample(sparse, k):
    """Keep every k-th annotated position per rip video (first annotated frame always kept); drop the other
    positions from the video (file_names/length) and from every track (segmentations/bboxes/areas)."""
    tracks_by_video = {}
    for a in sparse["annotations"]:
        tracks_by_video.setdefault(a["video_id"], []).append(a)
    videos, anns, anchors = [], [], {}
    for v in sparse["videos"]:
        stem = VIDEO_RE.match(v["file_names"][0]).group(2)
        tracks = tracks_by_video.get(v["id"], [])
        if "-NR-" in stem or not tracks:
            videos.append(v); anns.extend(tracks); continue
        annotated = [any(tr["segmentations"][p] is not None for tr in tracks) for p in range(v["length"])]
        ann_pos = [p for p, a in enumerate(annotated) if a]
        keep_pos = set(ann_pos[::k]) | {p for p, a in enumerate(annotated) if not a}   # unannotated positions stay
        keep = sorted(keep_pos)
        nv = dict(v, length=len(keep), file_names=[v["file_names"][p] for p in keep])
        videos.append(nv)
        for tr in tracks:
            anns.append(dict(tr, segmentations=[tr["segmentations"][p] for p in keep],
                             bboxes=[tr["bboxes"][p] for p in keep], areas=[tr["areas"][p] for p in keep]))
        anchors[stem] = sorted(int(VIDEO_RE.match(v["file_names"][p]).group(3)) for p in ann_pos[::k])
    return dict(sparse, videos=videos, annotations=anns), anchors


def main():
    bdv = load_densifier()
    sparse = json.load(open(SPARSE_JSON))
    n_ann = sum(1 for a in sparse["annotations"] for s in a["segmentations"] if s is not None)
    print(f"sparse: {len(sparse['videos'])} videos, {len(sparse['annotations'])} tracks, {n_ann} annotated masks", flush=True)
    for k in ANCHOR_STRIDES:
        out_dir = os.path.join(OUT_ROOT, f"ytvis_dense_interp_a{k}")
        if os.path.isfile(os.path.join(out_dir, "train.json")):
            print(f"a{k}: exists, skip", flush=True); continue
        t0 = time.time()
        sub, anchors = subsample(sparse, k)
        n_sub = sum(1 for a in sub["annotations"] for s in a["segmentations"] if s is not None)
        dense, manifest, n_dense = bdv.densify(sub, interp=True)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(dense, open(os.path.join(out_dir, "train.json"), "w"))
        with open(os.path.join(out_dir, f"anchors_a{k}.txt"), "w") as f:
            for stem, idx in sorted(anchors.items()):
                f.write(stem + "\t" + ",".join(map(str, idx)) + "\n")
        print(f"a{k}: {n_sub} anchor masks ({n_sub / n_ann:.3f} of the expert masks), {n_dense} interpolated frames, "
              f"{len(manifest)} videos, {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
