#!/usr/bin/env python3
"""Aggregate the student rows by VARIANT (run name minus `_s<seed>` and `_best_e<N>`):
mean +- std over seeds of F2@50 / AP50 / AP50-95 on both GT views (test), n seeds, best
epochs and samples seen. Reads results/student_eval_{bfi,tight}.tsv + student_train_summary.tsv,
writes results/student_summary_by_variant.tsv and prints a markdown table. No CLI args."""
import csv
import os
import re
from statistics import mean, pstdev

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "student_summary_by_variant.tsv")
VARIANT_RE = re.compile(r"^(?P<variant>.+?_(?:n|s))_s(?P<seed>\d+)_best_e(?P<epoch>\d+)$")  # <variant>_<size n|s>_s<seed>


def rows(fn):
    with open(os.path.join(RES, fn)) as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["split"] == "test"]


def main():
    per = {}
    for gt, fn in (("bfi", "student_eval_bfi.tsv"), ("tight", "student_eval_tight.tsv")):
        seen = set()
        for r in rows(fn):
            m = VARIANT_RE.match(r["model"])
            if not m or r["model"] in seen:
                continue
            seen.add(r["model"])
            variant = re.sub(r"^t1s\d+_", "t1sX_", m["variant"])  # two-stage runs carry the init seed in the name
            v = per.setdefault(variant, {"seeds": {}, "epochs": {}})
            v["seeds"].setdefault(m["seed"], {})[gt] = (float(r["F2_50"]), float(r["AP50"]), float(r["AP50_95"]))
            v["epochs"][m["seed"]] = int(m["epoch"])
    samples = {}
    p = os.path.join(RES, "student_train_summary.tsv")
    if os.path.isfile(p):
        for r in csv.DictReader(open(p), delimiter="\t"):
            samples[r["name"]] = r.get("samples_seen", "")

    def ms(vals):
        return f"{mean(vals):.3f} ± {pstdev(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"

    lines, tsv = [], ["variant\tn_seeds\tseeds\tbest_epochs\tbfi_F2_mean\tbfi_F2_std\tbfi_AP50_mean\tbfi_AP5095_mean\ttight_F2_mean\ttight_F2_std\ttight_AP50_mean"]
    for variant, v in sorted(per.items()):
        seeds = sorted(v["seeds"], key=int)
        b = [v["seeds"][s]["bfi"] for s in seeds if "bfi" in v["seeds"][s]]
        t = [v["seeds"][s]["tight"] for s in seeds if "tight" in v["seeds"][s]]
        if not b:
            continue
        f2b, apb, ap95b = [x[0] for x in b], [x[1] for x in b], [x[2] for x in b]
        f2t, apt = [x[0] for x in t], [x[1] for x in t]
        ep = ",".join(str(v["epochs"][s]) for s in seeds)
        lines.append(f"| {variant} | {len(b)} ({','.join(seeds)}) | {ep} | {ms(f2b)} | {ms(apb)} | {ms(ap95b)} | {ms(f2t) if f2t else '-'} | {ms(apt) if apt else '-'} |")
        tsv.append("\t".join([variant, str(len(b)), ",".join(seeds), ep, f"{mean(f2b):.4f}", f"{pstdev(f2b):.4f}" if len(f2b) > 1 else "",
                              f"{mean(apb):.4f}", f"{mean(ap95b):.4f}", f"{mean(f2t):.4f}" if f2t else "", f"{pstdev(f2t):.4f}" if len(f2t) > 1 else "", f"{mean(apt):.4f}" if apt else ""]))
    open(OUT, "w").write("\n".join(tsv) + "\n")
    print("| variant | n seeds | best epochs | bfi F2@50 | bfi AP50 | bfi AP50-95 | tight F2@50 | tight AP50 |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(lines))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
