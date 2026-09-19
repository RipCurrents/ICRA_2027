#!/usr/bin/env python3
"""Build a yolo26 DETECTION training view from teacher candidates (ICRA tier T1:
prediction-level data distillation on the native-fps TRAIN frames).

View = the canonical s1.3 train split with EXPERT labels on every annotated
frame (bbox_from_instance = polygon bounding rects, the same convention as the
teacher's mask rects) + the un-annotated native-fps frames of the same TRAIN
videos with TEACHER labels:
  * rip videos: teacher candidates (teacher_predict_frames.py .npz, mask rects)
    temporally smoothed over a +-K_SECONDS window by IoU linking (score = mean
    over the window with unmatched frames counting 0, coordinates = score-
    weighted mean of the linked boxes), kept if smoothed score >= TAU, then
    per-frame NMS. Frames with no surviving box are DROPPED unless
    KEEP_EMPTY_RIP_FRAMES (a teacher miss must not become a negative).
  * no-rip videos: every frame is a negative (empty label) - the video-level
    label is known, so the teacher's false positives there are corrected.
  * STRIDE keeps every STRIDE-th native frame (density ablation); smoothing
    always uses all frames' candidates.
val/test manifests point at the same expert GT view (evaluation stays on real
annotations; the benchmark evaluator is used downstream, never ultralytics'
val numbers). Layout follows ultralytics' last-/images/->/labels/ rule with
two-level symlinks (images/{sampled_frames,dense_frames}, labels/sampled_frames
-> expert labels, labels/dense_frames/ = written here). The view is then
LOADER-VERIFIED through ultralytics' real YOLODataset scan (counts must match
the builder's). Constants below, no CLI args. Env: conda yoloV26.
"""
import csv
import json
import os
import shutil
import time

import numpy as np

# --- CONFIG -------------------------------------------------------------------
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
GT_VIEW = os.path.join(DATASET_ROOT, "labels", "bbox_from_instance", "yolo26")  # expert labels + manifests
FRAMES_DIR = "/home/user/RipBench/icra_edge_frames/dense_frames"
HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_MANIFEST = os.path.join(HERE, "manifests", "train_frames_manifest.tsv")
TEACHER_TAG = "rfdetr_seg_nano"
PRED_DIR = os.path.join("/mnt/linux/icra_edge/predictions", "teacher_" + TEACHER_TAG)
VIEWS_ROOT = "/home/user/RipBench/icra_edge_views"  # NVMe (label scan reads 200k small files)

TAU = 0.24          # teacher's val-selected F2 operating point (results/teacher_boxified_eval.tsv)
K_SECONDS = 0.25    # +- temporal window; 0 = no smoothing (single-frame teacher labels)
MATCH_IOU = 0.5     # linking IoU across frames
NMS_IOU = 0.5
STRIDE = 1          # keep dense frame n iff n % STRIDE == 0 (1 = every native frame)
BOX_SOURCE = "rect"  # "rect" = mask bounding rect (bfi convention) | "head" = detection-head box
KEEP_EMPTY_RIP_FRAMES = False
INCLUDE_NEGATIVE_VIDEOS = True
ONLY_VIDEOS = None  # smoke-test subset, e.g. ["RipBench-005"]; None = all train videos
TEACHER_ONLY_KEEP_EMPTY = True  # (c): teacher-empty frames of TEACHER_ONLY_VIDEOS are kept as negatives (True, honest label-free
                                # recipe: ~17 % wrong negatives on rip footage) or dropped (False, the (c-drop) discriminator)
TEACHER_ONLY_VIDEOS = None  # semi-supervised holdout, condition (c): set of TRULY UNLABELLED videos - the VIDEO-LEVEL
                            # label (rips/ vs no-rips/ folder) is NOT used for them: every frame goes through the teacher;
                            # frames with a confident box become positives, all other frames become NEGATIVES (kept).
                            # Without this, no-rip videos are dense negatives by their folder name alone (critic memo 2026-08-27).
GT_ONLY_VIDEOS = None  # semi-supervised holdout: set of videos whose EXPERT frames may enter the view (their other
                       # frames + every frame of the remaining videos get teacher labels); None = all. Load with
                       # {r["video"] for r in csv.DictReader(open("manifests/holdout_h50_seed0.tsv"), delimiter="\t") if r["role"] == "labelled"}
VIEW_NAME = None    # None = auto: t1_<teacher>_k<K>_t<TAU>_s<STRIDE>[_keepempty][_head]
VERIFY = True       # ultralytics YOLODataset scan + count assertions at the end
# ------------------------------------------------------------------------------


def auto_name():
    n = f"t1_{TEACHER_TAG}_k{K_SECONDS:g}_t{TAU:g}_s{STRIDE}"
    if KEEP_EMPTY_RIP_FRAMES:
        n += "_keepempty"
    if BOX_SOURCE == "head":
        n += "_head"
    if not INCLUDE_NEGATIVE_VIDEOS:
        n += "_noneg"
    if ONLY_VIDEOS:
        n += "_smoke"
    return n


def iou_matrix(a, b):
    """xyxy (K,4) vs (M,4) -> (K,M)."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < thr]
    return np.array(keep, dtype=int)


def smooth_video(per_frame, n_frames, k):
    """per_frame: {n: (scores (K,), boxes (K,4))}. Returns {n: (scores, boxes)}
    after +-k window smoothing (frames outside [0, n_frames) are not in the
    window; unmatched in-range frames count as score 0)."""
    out = {}
    for t, (s_t, b_t) in per_frame.items():
        if k == 0:
            out[t] = (s_t, b_t)
            continue
        lo, hi = max(0, t - k), min(n_frames - 1, t + k)
        n_win = hi - lo + 1
        acc_s = s_t.copy()
        acc_b = b_t * s_t[:, None]
        acc_w = s_t.copy()
        for u in range(lo, hi + 1):
            if u == t or u not in per_frame:
                continue
            s_u, b_u = per_frame[u]
            ious = iou_matrix(b_t, b_u)
            j = ious.argmax(1)
            ok = ious[np.arange(len(b_t)), j] >= MATCH_IOU
            m = np.where(ok, s_u[j], 0.0)
            acc_s += m
            acc_b += b_u[j] * m[:, None]
            acc_w += m
        s_sm = acc_s / n_win
        b_sm = acc_b / np.maximum(acc_w, 1e-9)[:, None]
        out[t] = (s_sm, b_sm)
    return out


def load_candidates(stem):
    z = np.load(os.path.join(PRED_DIR, stem + ".npz"))
    c = z["cands"]
    W, H, n_frames = int(z["width"]), int(z["height"]), int(z["n_frames"])
    box_cols = slice(2, 6) if BOX_SOURCE == "rect" else slice(6, 10)
    per_frame = {}
    if len(c):
        order = np.argsort(c[:, 0], kind="stable")
        c = c[order]
        for n in np.unique(c[:, 0]).astype(int):
            rows = c[c[:, 0] == n]
            per_frame[n] = (rows[:, 1].astype(np.float64), rows[:, box_cols].astype(np.float64))
    return per_frame, W, H, n_frames, str(z["teacher"])


def write_label(path, boxes_xyxy, W, H):
    lines = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, x2 = np.clip([x1, x2], 0, W)
        y1, y2 = np.clip([y1, y2], 0, H)
        w, h = x2 - x1, y2 - y1
        if w < 2 or h < 2:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {w / W:.6f} {h / H:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def gt_frames():
    """{stem: set(native idx)} of annotated TRAIN frames (canonical, no additional_data)."""
    out = {}
    for line in open(os.path.join(GT_VIEW, "train.txt")):
        line = line.strip()
        if not line or "sampled_frames" not in line:
            continue
        base = os.path.basename(line)
        stem, idx = base.rsplit("_", 1)
        if GT_ONLY_VIDEOS is not None and stem not in GT_ONLY_VIDEOS:
            continue  # holdout: this video's expert labels are withheld -> all its frames are teacher-labelled
        out.setdefault(stem, set()).add(int(idx[:-4]))
    return out


def rewrite_manifest(src, dst, view):
    n = 0
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            assert line.startswith("./images/sampled_frames/"), line
            fo.write(os.path.join(view, "images", line[len("./images/"):]) + "\n")
            n += 1
    return n


def verify_view(view, expected):
    """Loader verification through ultralytics' real dataset scan."""
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(os.path.join(view, "data.yaml"))
    ds = YOLODataset(img_path=data["train"], data=data, task="detect", imgsz=640,
                     augment=False, batch_size=8, cache=False)
    n_img = len(ds.labels)
    n_boxes = int(sum(len(lb["cls"]) for lb in ds.labels))
    n_nonempty = int(sum(len(lb["cls"]) > 0 for lb in ds.labels))
    print(f"[verify] ultralytics scan: {n_img} images, {n_nonempty} with boxes, {n_boxes} boxes")
    exp_img = expected["n_gt_frames"] + expected["n_dense_frames"]
    exp_boxes = expected["n_gt_boxes"] + expected["n_dense_boxes"]
    assert n_img == exp_img, (n_img, exp_img)
    assert n_boxes == exp_boxes, (n_boxes, exp_boxes)
    # sample a few dense frames end-to-end (image decode + label parse)
    dense_idx = [i for i, f in enumerate(ds.im_files) if "/dense_frames/" in f][:3]
    for i in dense_idx:
        item = ds[i]
        print(f"[verify] {os.path.basename(ds.im_files[i])}: img {tuple(item['img'].shape)}, "
              f"{len(item['cls'])} boxes")
    print("[verify] OK")


def main():
    t0 = time.time()
    view = os.path.join(VIEWS_ROOT, VIEW_NAME or auto_name())
    if os.path.isdir(view):
        shutil.rmtree(view)
    os.makedirs(os.path.join(view, "images"))
    os.makedirs(os.path.join(view, "labels", "dense_frames"))
    os.symlink(os.path.join(DATASET_ROOT, "images", "sampled_frames"),
               os.path.join(view, "images", "sampled_frames"))
    os.symlink(FRAMES_DIR, os.path.join(view, "images", "dense_frames"))
    # labels/sampled_frames is a REAL dir with class-level symlinks: ultralytics writes its
    # scan .cache next to the first label's parent dir -> it must land in the view, never in
    # the shared dataset label tree
    os.makedirs(os.path.join(view, "labels", "sampled_frames"))
    for cls in ("rips", "no-rips"):
        os.symlink(os.path.join(GT_VIEW, "labels", "sampled_frames", cls),
                   os.path.join(view, "labels", "sampled_frames", cls))

    with open(FRAMES_MANIFEST) as f:
        vids = {r["stem"]: r for r in csv.DictReader(f, delimiter="\t") if r["check_ok"] == "1"}
    gt = gt_frames()
    assert set(gt) <= set(vids) or ONLY_VIDEOS, sorted(set(gt) - set(vids))[:5]
    stems = sorted(vids) if not ONLY_VIDEOS else [s for s in sorted(vids) if s in ONLY_VIDEOS]

    dense_lines, stats = [], {}
    n_dense_boxes = n_dense_frames = 0
    teacher_name = None
    for stem in stems:
        r = vids[stem]
        n_frames = int(r["n_extracted"])
        teacher_only = TEACHER_ONLY_VIDEOS is not None and stem in TEACHER_ONLY_VIDEOS
        is_rip = r["video_class"] == "rip" or teacher_only          # teacher_only: folder label ignored
        keep_empty = KEEP_EMPTY_RIP_FRAMES or (teacher_only and TEACHER_ONLY_KEEP_EMPTY)   # (c): teacher-empty frames = negatives
        gt_idx = gt.get(stem, set())
        dense_idx = [n for n in range(n_frames) if n not in gt_idx and n % STRIDE == 0]
        kept = boxes_written = 0
        if is_rip:
            per_frame, W, H, nf, teacher_name = load_candidates(stem)
            assert nf == n_frames and W == int(r["out_width"]) and H == int(r["out_height"]), stem
            k = int(round(K_SECONDS * float(r["fps"])))
            sm = smooth_video(per_frame, n_frames, k)
            for n in dense_idx:
                s, b = sm.get(n, (np.zeros(0), np.zeros((0, 4))))
                m = s >= TAU
                s, b = s[m], b[m]
                if len(s):
                    ki = nms(b, s, NMS_IOU)
                    b = b[ki]
                nb = write_label(os.path.join(view, "labels", "dense_frames", f"{stem}_{n:05d}.txt"), b, W, H) if len(b) else 0
                if nb == 0 and not keep_empty:
                    p = os.path.join(view, "labels", "dense_frames", f"{stem}_{n:05d}.txt")
                    if os.path.isfile(p):
                        os.remove(p)
                    continue
                if nb == 0:
                    open(os.path.join(view, "labels", "dense_frames", f"{stem}_{n:05d}.txt"), "w").close()
                dense_lines.append(os.path.join(view, "images", "dense_frames", f"{stem}_{n:05d}.jpg"))
                kept += 1
                boxes_written += nb
        elif INCLUDE_NEGATIVE_VIDEOS:
            for n in dense_idx:
                open(os.path.join(view, "labels", "dense_frames", f"{stem}_{n:05d}.txt"), "w").close()
                dense_lines.append(os.path.join(view, "images", "dense_frames", f"{stem}_{n:05d}.jpg"))
                kept += 1
        stats[stem] = dict(video_class=r["video_class"], teacher_only=teacher_only, n_frames=n_frames, n_gt=len(gt_idx),
                           n_dense_candidates=len(dense_idx), n_dense_kept=kept, n_dense_boxes=boxes_written)
        n_dense_boxes += boxes_written
        n_dense_frames += kept
        print(f"  {stem} ({r['video_class']}): {n_frames} frames, {len(gt_idx)} GT, "
              f"{kept}/{len(dense_idx)} dense kept, {boxes_written} pseudo boxes", flush=True)

    # manifests: dense lines FIRST (ultralytics names the split cache after the first label's
    # parent dir -> train = labels/dense_frames.cache, val/test = labels/sampled_frames.cache)
    n_gt_frames = rewrite_manifest(os.path.join(GT_VIEW, "train.txt"), os.path.join(view, "gt_train.txt"), view)
    if ONLY_VIDEOS or GT_ONLY_VIDEOS is not None:
        allowed = set(ONLY_VIDEOS or []) | set(GT_ONLY_VIDEOS or []) if not (ONLY_VIDEOS and GT_ONLY_VIDEOS is not None) \
            else set(ONLY_VIDEOS) & set(GT_ONLY_VIDEOS)
        keep = [l for l in open(os.path.join(view, "gt_train.txt")) if any(f"/{s}_" in l for s in allowed)]
        open(os.path.join(view, "gt_train.txt"), "w").writelines(keep)
        n_gt_frames = len(keep)
    with open(os.path.join(view, "train.txt"), "w") as f:
        f.write("\n".join(dense_lines) + "\n")
        f.write(open(os.path.join(view, "gt_train.txt")).read())
    for split in ("val", "test"):
        rewrite_manifest(os.path.join(GT_VIEW, f"{split}.txt"), os.path.join(view, f"{split}.txt"), view)
    with open(os.path.join(view, "data.yaml"), "w") as f:
        f.write(f"path: {view}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: rip_current\n")
    # expert boxes on the GT frames (for the verification count)
    n_gt_boxes = 0
    for line in open(os.path.join(view, "gt_train.txt")):
        lp = line.strip().replace("/images/", "/labels/", 1)[:-4] + ".txt"
        if os.path.isfile(lp):
            n_gt_boxes += sum(1 for l in open(lp) if l.strip())
    info = dict(view=view, teacher_tag=TEACHER_TAG, teacher_ckpt=teacher_name, pred_dir=PRED_DIR,
                gt_view=GT_VIEW, tau=TAU, k_seconds=K_SECONDS, match_iou=MATCH_IOU, nms_iou=NMS_IOU,
                stride=STRIDE, box_source=BOX_SOURCE, keep_empty_rip_frames=KEEP_EMPTY_RIP_FRAMES,
                include_negative_videos=INCLUDE_NEGATIVE_VIDEOS, only_videos=ONLY_VIDEOS,
                gt_only_videos=(sorted(GT_ONLY_VIDEOS) if GT_ONLY_VIDEOS is not None else None),
                teacher_only_videos=(sorted(TEACHER_ONLY_VIDEOS) if TEACHER_ONLY_VIDEOS is not None else None),
                teacher_only_keep_empty=TEACHER_ONLY_KEEP_EMPTY,
                n_videos=len(stems), n_gt_frames=n_gt_frames, n_gt_boxes=n_gt_boxes,
                n_dense_frames=n_dense_frames, n_dense_boxes=n_dense_boxes,
                n_train_images=n_gt_frames + n_dense_frames, per_video=stats,
                built=time.strftime("%Y-%m-%d %H:%M"), build_seconds=round(time.time() - t0))
    with open(os.path.join(view, "build_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"VIEW {view}: {n_gt_frames} GT frames ({n_gt_boxes} expert boxes) + {n_dense_frames} dense frames "
          f"({n_dense_boxes} pseudo boxes) = {n_gt_frames + n_dense_frames} train images "
          f"[{time.time() - t0:.0f}s]", flush=True)
    if VERIFY:
        verify_view(view, info)


if __name__ == "__main__":
    main()
