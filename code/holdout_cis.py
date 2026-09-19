#!/usr/bin/env python3
"""Paired video-level CIs for the semi-supervised holdout arms (runs bootstrap_cis with GROUP_PAIRS derived from the
results TSVs, so no stems have to be typed by hand once new arms land).

Arms are matched by run-name prefix in results/student_eval_bfi.tsv (test rows): (a) h50_sparse_n, (b) h50_interp_n,
(b+) h50_interp_negall_n, (c) h50_teacher_n, (c-drop) h50_teacher_dropempty_n, canonical canon_bfi_n, and the yolo26s
pair h50_sparse_s / h50_interp_s. Pairs below; missing arms are skipped. Output rows -> results/student_bootstrap_cis.tsv
(both GT views, seed-pooled). Constants below, no CLI args. Env: conda yoloV26 (CPU).
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bootstrap_cis as bc  # noqa: E402

ARMS = {"canon_n": r"^canon_bfi_n_s\d+_best_e\d+$", "h50_sparse_n": r"^h50_sparse_n_s\d+_best_e\d+$",
        "h50_interp_n": r"^h50_interp_n_s\d+_best_e\d+$", "h50_interp_negall_n": r"^h50_interp_negall_n_s\d+_best_e\d+$",
        "h50_teacher_n": r"^h50_teacher_n_s\d+_best_e\d+$", "h50_teacher_dropempty_n": r"^h50_teacher_dropempty_n_s\d+_best_e\d+$",
        "h50_sparse_s": r"^h50_sparse_s_s\d+_best_e\d+$", "h50_interp_s": r"^h50_interp_s_s\d+_best_e\d+$"}
PAIRS = [("h50_interp_negall_n", "h50_teacher_n"), ("h50_interp_n", "h50_teacher_n"), ("h50_sparse_n", "h50_teacher_n"),
         ("h50_teacher_n", "h50_teacher_dropempty_n"), ("h50_interp_negall_n", "h50_teacher_dropempty_n"),
         ("canon_n", "h50_teacher_n"), ("h50_sparse_n", "h50_interp_n"), ("h50_interp_n", "h50_interp_negall_n"),
         ("h50_sparse_s", "h50_interp_s")]


def main():
    stems = {}
    for r in csv.DictReader(open(os.path.join(HERE, "results", "student_eval_bfi.tsv")), delimiter="\t"):
        if r["split"] != "test":
            continue
        for arm, rx in ARMS.items():
            if re.match(rx, r["model"]):
                stems.setdefault(arm, set()).add(r["model"])
    groups = []
    for a, b in PAIRS:
        if a in stems and b in stems:
            groups.append(((a, sorted(stems[a])), (b, sorted(stems[b]))))
            print(f"pair {b} - {a}: {len(stems[b])} vs {len(stems[a])} seeds")
        else:
            print(f"skip {b} - {a}: missing arm(s)")
    bc.GROUP_PAIRS = groups
    bc.PAIRS = []
    bc.main()


if __name__ == "__main__":
    main()
