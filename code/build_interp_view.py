#!/usr/bin/env python3
"""Teacher-FREE densification control view (ICRA T1 table row 3).

Same layout / GT frames / negatives as build_teacher_view.py, but the rip-video
dense frames get boxes from the INTERPOLATED EXPERT MASKS of the video-model
work (labels/video_instance_segmentation/ytvis_dense_interp/train.json: SDF
interpolation between consecutive annotated frames, gate-validated median IoU
0.91-0.99 vs held-out annotations; only frames strictly BETWEEN two annotated
frames exist there - never beyond a video's first/last annotation). Boxes are
the YTVIS per-frame bboxes (xywh, native pixels) normalised by the video's
native size -> valid for the 1280-px extracted frames. No-rip videos get every
native frame as a negative exactly like the teacher view (INCLUDE_NEGATIVE_VIDEOS),
so the ONLY difference to the teacher view is who labels the rip-video frames
(and that the teacher also covers frames outside annotated pairs).
Loader-verified through ultralytics' YOLODataset. Env: conda yoloV26.
"""
import importlib.util
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("btv", os.path.join(HERE, "build_teacher_view.py"))
btv = importlib.util.module_from_spec(spec)
sys.modules["btv"] = btv
spec.loader.exec_module(btv)

# --- CONFIG -------------------------------------------------------------------
DATASET_ROOT = btv.DATASET_ROOT
INTERP_JSON = os.path.join(DATASET_ROOT, "labels", "video_instance_segmentation",
                           "ytvis_dense_interp", "train.json")
STRIDE = 1
INCLUDE_NEGATIVE_VIDEOS = True
VIEW_NAME = f"interp_gt_s{STRIDE}"
VERIFY = True
GT_ONLY_FRAMES = None  # label-spacing sweep: {stem: set(native idx)} of the expert frames that KEEP their labels (anchors);
                       # every other sampled frame is treated as un-annotated (its interpolated label comes from INTERP_JSON)
# ------------------------------------------------------------------------------
FN_RE = re.compile(r"^(?:dense_frames|sampled_frames/(?:rips|no-rips))/(RipBench-(?:NR-)?\d+)_(\d+)\.jpg$")


def main():
    t0 = time.time()
    view = os.path.join(btv.VIEWS_ROOT, VIEW_NAME)
    if os.path.isdir(view):
        shutil.rmtree(view)
    os.makedirs(os.path.join(view, "images"))
    os.makedirs(os.path.join(view, "labels", "dense_frames"))
    os.makedirs(os.path.join(view, "labels", "sampled_frames"))
    os.symlink(os.path.join(DATASET_ROOT, "images", "sampled_frames"), os.path.join(view, "images", "sampled_frames"))
    os.symlink(btv.FRAMES_DIR, os.path.join(view, "images", "dense_frames"))
    for cls in ("rips", "no-rips"):
        os.symlink(os.path.join(btv.GT_VIEW, "labels", "sampled_frames", cls),
                   os.path.join(view, "labels", "sampled_frames", cls))
    import csv
    with open(btv.FRAMES_MANIFEST) as f:
        vids = {r["stem"]: r for r in csv.DictReader(f, delimiter="\t") if r["check_ok"] == "1"}
    gt = btv.gt_frames()
    if GT_ONLY_FRAMES is not None:
        gt = {st: (fr & GT_ONLY_FRAMES.get(st, set())) for st, fr in gt.items()}
    d = json.load(open(INTERP_JSON))
    tracks = {}
    for a in d["annotations"]:
        tracks.setdefault(a["video_id"], []).append(a)
    dense_lines, stats = [], {}
    n_dense_frames = n_dense_boxes = 0
    seen_stems = set()
    for v in d["videos"]:
        stem = FN_RE.match(v["file_names"][0]).group(1)
        seen_stems.add(stem)
        r = vids[stem]
        W, H = v["width"], v["height"]
        # compare against the EXTRACTED frame size (ffprobe reports coded dims; rotated portrait
        # videos are 1080x1920 after the rotation ffmpeg and the canonical JPEGs both apply)
        ow, oh = int(r["out_width"]), int(r["out_height"])
        assert abs(W / H - ow / oh) < 0.01, (stem, W, H, ow, oh)
        kept = boxes = 0
        for i, fn in enumerate(v["file_names"]):
            if not fn.startswith("dense_frames/"):
                continue
            n = int(FN_RE.match(fn).group(2))
            assert n not in gt.get(stem, set()) and n < int(r["n_extracted"]), (stem, n)
            if n % STRIDE:
                continue
            bx = [a["bboxes"][i] for a in tracks.get(v["id"], []) if a["bboxes"][i] is not None]
            xyxy = [[x, y, x + w, y + h] for x, y, w, h in bx]
            nb = btv.write_label(os.path.join(view, "labels", "dense_frames", f"{stem}_{n:05d}.txt"), xyxy, W, H)
            dense_lines.append(os.path.join(view, "images", "dense_frames", f"{stem}_{n:05d}.jpg"))
            kept += 1
            boxes += nb
        stats[stem] = dict(video_class=r["video_class"], n_dense_kept=kept, n_dense_boxes=boxes)
        n_dense_frames += kept
        n_dense_boxes += boxes
    if INCLUDE_NEGATIVE_VIDEOS:
        for stem, r in sorted(vids.items()):
            if r["video_class"] != "no_rip":
                continue
            kept = 0
            for n in range(int(r["n_extracted"])):
                if n in gt.get(stem, set()) or n % STRIDE:
                    continue
                open(os.path.join(view, "labels", "dense_frames", f"{stem}_{n:05d}.txt"), "w").close()
                dense_lines.append(os.path.join(view, "images", "dense_frames", f"{stem}_{n:05d}.jpg"))
                kept += 1
            stats[stem] = dict(video_class="no_rip", n_dense_kept=kept, n_dense_boxes=0)
            n_dense_frames += kept
    n_gt_frames = btv.rewrite_manifest(os.path.join(btv.GT_VIEW, "train.txt"), os.path.join(view, "gt_train.txt"), view)
    if GT_ONLY_FRAMES is not None:
        keep = []
        for l in open(os.path.join(view, "gt_train.txt")):
            m = FN_RE.match("sampled_frames/" + l.strip().split("/images/sampled_frames/")[1]) if "/images/sampled_frames/" in l else None
            st, n = (m.group(1), int(m.group(2))) if m else (None, None)
            if st is None or "NR-" in st or n in GT_ONLY_FRAMES.get(st, set()):
                keep.append(l)
        open(os.path.join(view, "gt_train.txt"), "w").writelines(keep)
        n_gt_frames = len(keep)
    with open(os.path.join(view, "train.txt"), "w") as f:
        f.write("\n".join(dense_lines) + "\n")
        f.write(open(os.path.join(view, "gt_train.txt")).read())
    for split in ("val", "test"):
        btv.rewrite_manifest(os.path.join(btv.GT_VIEW, f"{split}.txt"), os.path.join(view, f"{split}.txt"), view)
    with open(os.path.join(view, "data.yaml"), "w") as f:
        f.write(f"path: {view}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: rip_current\n")
    n_gt_boxes = 0
    for line in open(os.path.join(view, "gt_train.txt")):
        lp = line.strip().replace("/images/", "/labels/", 1)[:-4] + ".txt"
        if os.path.isfile(lp):
            n_gt_boxes += sum(1 for l in open(lp) if l.strip())
    info = dict(view=view, source=INTERP_JSON, stride=STRIDE, include_negative_videos=INCLUDE_NEGATIVE_VIDEOS,
                n_videos=len(seen_stems), n_gt_frames=n_gt_frames, n_gt_boxes=n_gt_boxes,
                n_dense_frames=n_dense_frames, n_dense_boxes=n_dense_boxes,
                n_train_images=n_gt_frames + n_dense_frames, per_video=stats,
                built=time.strftime("%Y-%m-%d %H:%M"), build_seconds=round(time.time() - t0))
    json.dump(info, open(os.path.join(view, "build_info.json"), "w"), indent=1)
    print(f"VIEW {view}: {n_gt_frames} GT frames ({n_gt_boxes} expert boxes) + {n_dense_frames} dense frames "
          f"({n_dense_boxes} interpolated boxes) = {n_gt_frames + n_dense_frames} train images", flush=True)
    if VERIFY:
        btv.verify_view(view, info)


if __name__ == "__main__":
    main()
