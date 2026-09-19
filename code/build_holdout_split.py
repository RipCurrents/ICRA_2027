#!/usr/bin/env python3
"""Video-level holdout of expert labels inside the s1.3 TRAIN split (semi-supervised /
data-distillation control for the ICRA paper, design: ICRA_2027/notes/design_semisup_holdout.md).

Every train video is assigned to LABELLED (its expert labels may be used by the
teacher, the sparse student and the interpolated view) or UNLABELLED (its expert
labels must never be read; only the raw frames exist for the teacher to label).
Stratified by rip/no-rip, seeded, deterministic. Val/test are untouched.
Output: manifests/holdout_h<FRAC>_seed<SEED>.tsv (video, kind, role, n_frames).
Constants below, no CLI args.
"""
import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_MANIFEST = os.path.join(HERE, "manifests", "train_frames_manifest.tsv")
FRAC_LABELLED = 0.5
SEED = 0
OUT = os.path.join(HERE, "manifests", f"holdout_h{int(FRAC_LABELLED * 100)}_seed{SEED}.tsv")


def main():
    rows = list(csv.DictReader(open(FRAMES_MANIFEST), delimiter="\t"))
    vid_col, nf_col = "stem", "n_extracted"          # train_frames_manifest.tsv columns
    videos = sorted({r[vid_col] for r in rows})
    nframes = {r[vid_col]: int(r[nf_col]) for r in rows}
    assert len(videos) == 165, len(videos)
    rng = random.Random(SEED)
    out = []
    for kind in ("rip", "no-rip"):
        group = [v for v in videos if ("NR-" in v) == (kind == "no-rip")]
        rng.shuffle(group)
        n_lab = round(FRAC_LABELLED * len(group))
        for i, v in enumerate(sorted(group[:n_lab])):
            out.append((v, kind, "labelled", nframes[v]))
        for v in sorted(group[n_lab:]):
            out.append((v, kind, "unlabelled", nframes[v]))
    out.sort()
    with open(OUT, "w") as f:
        f.write("video\tkind\trole\tn_native_frames\n")
        for r in out:
            f.write("\t".join(map(str, r)) + "\n")
    for role in ("labelled", "unlabelled"):
        for kind in ("rip", "no-rip"):
            sel = [r for r in out if r[2] == role and r[1] == kind]
            print(f"{role:10s} {kind:6s}: {len(sel):3d} videos, {sum(r[3] for r in sel):7d} native frames")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
