#!/usr/bin/env python3
"""Teacher inference over ALL extracted native-fps TRAIN frames -> per-video
box candidates for pseudo-labelling (ICRA edge distillation, tier T1).

Teacher = an RF-DETR-Seg checkpoint from the benchmark (cluster best-ema
.pth). Re-implements rfdetr's predict() preprocessing exactly (to_tensor ->
torchvision resize to the square model resolution, antialiased bilinear ->
ImageNet normalize) but does the post-processing itself: sigmoid + top-k over
queries x classes, keep the rip class only (label 0; rfdetr's extra logit is
the background class), keep candidates with score >= SCORE_FLOOR (max
MAX_CAND), and box-ify ONLY those masks on the GPU (bilinear upsample to the
frame size, > 0, row/col any -> bounding rect = the bbox_from_instance
convention used by the benchmark teacher rows). The detection-head box is
stored as well so the two conventions can be compared. Nothing full-res
leaves the GPU, which is what made the benchmark RLE dumper slow.

Per video one .npz in OUT_DIR: cands float32 (N, 10) = [frame_idx, score,
rx1, ry1, rx2, ry2 (mask rect), bx1, by1, bx2, by2 (head box)] in EXTRACTED-
frame pixel coords + meta (stem, width, height, n_frames, teacher, floor).
Frames come from extract_train_frames.py (manifest rows with check_ok=1 only).
Resumable (existing .npz skipped). Constants below, no CLI args.
Env: conda rfdetr (rfdetr 1.6.5, torch 2.5.1). Coexists with CPU work only -
one GPU job at a time on the shared 4090.
"""
import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

# --- CONFIG -------------------------------------------------------------------
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
TEACHER_CKPT = os.environ.get("EDGE_TEACHER_CKPT") or os.path.join(
    DATASET_ROOT, "models", "instance_segmentation", "best_models",
    "1_seg_rfdetr_nano_best_best_ema_j2995976.pth")
TEACHER_TAG = os.environ.get("EDGE_TEACHER_TAG", "rfdetr_seg_nano")  # canonical teacher = the clean story;
# ablation teachers via env: EDGE_TEACHER_CKPT=<.pth> EDGE_TEACHER_TAG=<tag> (e.g. rfdetr_seg_nano_add)
FRAMES_DIR = "/home/user/RipBench/icra_edge_frames/dense_frames"
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifests", "train_frames_manifest.tsv")
OUT_DIR = os.path.join("/mnt/linux/icra_edge/predictions", "teacher_" + TEACHER_TAG)
SCORE_FLOOR = 0.02  # val-optimal operating points are ~0.24; the temporal smoother needs the tail
MAX_CAND = 20       # candidates kept per frame (rips per frame are 1-4)
BATCH = 32
WORKERS = int(os.environ.get("EDGE_WORKERS", "8"))
ONLY_VIDEOS = None  # e.g. ["RipBench-005"] for a smoke test; None = every manifest video
SKIP_EXISTING = True
# ------------------------------------------------------------------------------

SIZE_CLASS = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall", "medium": "RFDETRSegMedium",
              "large": "RFDETRSegLarge", "xlarge": "RFDETRSegXLarge", "2xlarge": "RFDETRSeg2XLarge"}
MEANS, STDS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class FrameSet(Dataset):
    def __init__(self, stem, n_frames):
        self.stem, self.n = stem, n_frames

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        p = os.path.join(FRAMES_DIR, f"{self.stem}_{i:05d}.jpg")
        img = cv2.imread(p)
        assert img is not None, p
        return i, torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # HWC uint8


def load_teacher():
    import rfdetr as rfdetr_pkg
    stem = os.path.basename(TEACHER_CKPT)
    size = next(p for p in stem.split("_") if p in SIZE_CLASS)
    cls = getattr(rfdetr_pkg, SIZE_CLASS[size])
    model = cls(pretrain_weights=TEACHER_CKPT, num_classes=1)
    net = model.model.model.eval().to(torch.device("cuda"))
    res = model.model.resolution
    print(f"teacher {stem} -> {cls.__name__} @ {res}x{res}, queries {net.num_queries}", flush=True)
    return net, res


@torch.no_grad()
def infer_batch(net, res, imgs_u8, device):
    """imgs_u8: (B, H, W, 3) uint8 on device. Returns per-image candidate arrays."""
    B, H, W, _ = imgs_u8.shape
    x = imgs_u8.permute(0, 3, 1, 2).float().div_(255.0)          # = torchvision to_tensor
    x = TF.resize(x, [res, res])                                  # bilinear, antialias (rfdetr predict path)
    x = TF.normalize(x, MEANS, STDS)
    out = net(x)
    if isinstance(out, tuple):
        out = {"pred_boxes": out[0], "pred_logits": out[1], **({"pred_masks": out[2]} if len(out) == 3 else {})}
    logits, boxes, masks = out["pred_logits"], out["pred_boxes"], out.get("pred_masks")
    C = logits.shape[2]
    prob = logits.sigmoid()[..., 0]  # rip class only (index C-1 is rfdetr's background logit)
    scores, qidx = prob.sort(dim=1, descending=True)
    scores, qidx = scores[:, :MAX_CAND], qidx[:, :MAX_CAND]
    cx, cy, bw, bh = boxes.unbind(-1)
    xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
    xyxy = xyxy * torch.tensor([W, H, W, H], device=device, dtype=xyxy.dtype)
    results = []
    for i in range(B):
        keep = scores[i] >= SCORE_FLOOR
        s, q = scores[i][keep], qidx[i][keep]
        if len(s) == 0:
            results.append(np.zeros((0, 9), np.float32))
            continue
        hb = xyxy[i][q]                                           # head boxes (K, 4)
        if masks is not None:
            m = masks[i][q].unsqueeze(1)                          # (K, 1, hm, wm) logits
            m = Fnn.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)[:, 0] > 0
            rows, cols = m.any(dim=2), m.any(dim=1)               # (K, H), (K, W)
            nonempty = rows.any(dim=1)
            ar = torch.arange(H, device=device)
            ac = torch.arange(W, device=device)
            y1 = torch.where(rows, ar, H).min(dim=1).values
            y2 = torch.where(rows, ar, -1).max(dim=1).values + 1
            x1 = torch.where(cols, ac, W).min(dim=1).values
            x2 = torch.where(cols, ac, -1).max(dim=1).values + 1
            rect = torch.stack([x1, y1, x2, y2], -1).float()
            s, rect, hb = s[nonempty], rect[nonempty], hb[nonempty]
        else:
            rect = hb
        results.append(torch.cat([s[:, None], rect, hb], 1).cpu().numpy().astype(np.float32))
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST) as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["check_ok"] == "1"]
    if ONLY_VIDEOS:
        rows = [r for r in rows if r["stem"] in ONLY_VIDEOS]
    device = torch.device("cuda")
    net, res = load_teacher()
    t_all, n_all = time.time(), 0
    for r in sorted(rows, key=lambda r: r["stem"]):
        stem, n_frames = r["stem"], int(r["n_extracted"])
        out_path = os.path.join(OUT_DIR, stem + ".npz")
        if SKIP_EXISTING and os.path.isfile(out_path):
            continue
        W, H = int(r["out_width"]), int(r["out_height"])
        dl = DataLoader(FrameSet(stem, n_frames), batch_size=BATCH, num_workers=WORKERS,
                        shuffle=False, pin_memory=True)
        cands, t0 = [], time.time()
        for idx, imgs in dl:
            assert imgs.shape[1] == H and imgs.shape[2] == W, (stem, imgs.shape)
            res_b = infer_batch(net, res, imgs.to(device, non_blocking=True), device)
            for fi, c in zip(idx.tolist(), res_b):
                if len(c):
                    cands.append(np.concatenate([np.full((len(c), 1), fi, np.float32), c], 1))
        cands = np.concatenate(cands, 0) if cands else np.zeros((0, 10), np.float32)
        np.savez_compressed(out_path, cands=cands, stem=stem, width=W, height=H, n_frames=n_frames,
                            teacher=os.path.basename(TEACHER_CKPT), score_floor=SCORE_FLOOR,
                            max_cand=MAX_CAND, resolution=res)
        dt = time.time() - t0
        n_all += n_frames
        print(f"  {stem}: {n_frames} frames, {len(cands)} cands (>= {SCORE_FLOOR}), "
              f"{(cands[:, 1] >= 0.25).sum()} @>=0.25, {n_frames / dt:.0f} fps", flush=True)
    print(f"TEACHER_PREDICT_DONE {n_all} frames in {(time.time() - t_all) / 60:.1f} min -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
