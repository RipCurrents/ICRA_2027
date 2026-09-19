#!/usr/bin/env python3
"""Whole-unit camera windows (device-week, Codex): one row per board x scene x policy from the audited
completed_board_review/camera_rows.tsv (primary rows only) plus the two RPi 4 units and the normal FRDM board read
straight from their cases/*/result.json (same fields; their audits are the per-folder HEALTH/FINAL_COLLECTION audits).
Writes results/camera_policy_rows.tsv (source path per row) and the paper table ICRA_2027/paper/tables/camera_policy.tex.
Constants below, no CLI args."""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DR = os.path.join(REPO, "ICRA_2027", "device_results")
REVIEW = os.path.join(DR, "imx95_frdm_measurements_20260912", "completed_board_review", "camera_rows.tsv")
EXTRA = {  # board -> glob of result.json files (rip + no-rip folders)
    "pi4_heatsink": [os.path.join(DR, "pi4_heatsink_camera_20260911", "cases", "rip", "*", "result.json"),
                     os.path.join(DR, "pi4_heatsink_norip_20260912", "cases", "norip", "*", "result.json")],
    "pi4_bare": [os.path.join(DR, "pi4_bare_camera_20260912", "cases", "rip", "*", "result.json"),
                 os.path.join(DR, "pi4_bare_norip_20260912", "cases", "norip", "*", "result.json")],
    "frdm": [os.path.join(DR, "imx95_frdm_camera_*", "cases", "*", "*", "result.json"),
             os.path.join(DR, "imx95_frdm_norip_*", "cases", "*", "*", "result.json")],
}
POLICIES = ["detector_only", "cascade", "adaptive", "max_throughput"]
LABEL = {"orin": "Orin (7\\,W) / RF-DETR-n FP16", "nano": "Nano 2019 / yolo26s FP16", "nxp": "i.MX95 Pro / QAT-n",
         "pi5": "RPi~5 / RF-DETR-n CPU", "pi4_heatsink": "RPi~4 heatsink / yolo26n CPU", "pi4_bare": "RPi~4 bare / yolo26n CPU",
         "frdm": "i.MX95 FRDM / QAT-n"}
ORDER = ["orin", "nano", "nxp", "frdm", "pi5", "pi4_heatsink", "pi4_bare"]
OUT_TSV = os.path.join(HERE, "results", "camera_policy_rows.tsv")
OUT_TEX = os.path.join(REPO, "ICRA_2027", "paper", "tables", "camera_policy.tex")


def main():
    rows = []
    for r in csv.DictReader(open(REVIEW), delimiter="\t"):
        if r["role"] != "primary":
            continue
        rows.append(dict(board=r["board"], scene=r["scene"], policy=r["policy"], W=float(r["mean_W"]),
                         dec_s=float(r["decisions_per_s"] or 0), calls=int(r["detector_calls"] or 0),
                         positives=int(r["positive_detector_decisions"] or 0), source=r["result"], audit="completed_board_review"))
    # explicit acceptance records (Codex-test, e.g. paper_writing_*/PI4_CAMERA_ACCEPTANCE_V1.json): when a board has one, only its
    # listed result paths are used, with the recorded acceptance; directory order never selects between attempts.
    accept = {}
    for a in glob.glob(os.path.join(DR, "**", "*CAMERA_ACCEPTANCE_V*.json"), recursive=True):
        for r in json.load(open(a)).get("rows", []):
            accept[os.path.normpath(os.path.join(REPO, r["result"]))] = (r.get("acceptance", ""), os.path.relpath(a, DR))
    boards_with_record = set()
    for path in accept:
        for board in EXTRA:
            if any(os.path.normpath(path).startswith(os.path.normpath(os.path.join(DR, p.split("*")[0]))) for p in [os.path.basename(x.split("/cases")[0]) for x in EXTRA[board]] if p) or board in os.path.relpath(path, DR).replace("_camera", "").replace("_norip", "").split("/")[0]:
                boards_with_record.add(board)
    for board, globs in EXTRA.items():
        for g in globs:
            for f in sorted(glob.glob(g)):
                d = json.load(open(f))
                pol = os.path.basename(os.path.dirname(f)).split("_", 1)[1]
                key = os.path.normpath(f)
                if board in boards_with_record:
                    if key not in accept:
                        continue   # every unselected attempt stays in its original archive, outside the displayed cells
                    acceptance, acceptance_file = accept[key]
                    if acceptance.startswith("accepted"):
                        assert d.get("status") == "completed", (f, acceptance)
                        d["reporting_status"] = acceptance + " (" + acceptance_file + ")"
                    elif acceptance == "excluded_from_accepted_power_results":
                        d["status"] = "power_rejected"   # explicit owner-recorded classification, no text search
                    else:
                        continue   # warmup or any other non-result record
                if pol not in POLICIES or d.get("warmup_only"):
                    continue
                if d.get("status") == "power_rejected" or (d.get("status") == "failed" and "gap" in json.dumps(d).lower()):     # ran, power record rejected: rej.
                    rows.append(dict(board=board, scene=d["phase"], policy=pol, W=None, dec_s=0.0, calls=d.get("detector_calls", 0),
                                     positives=d.get("alerts", 0), source=os.path.relpath(f, DR), audit="power_rejected"))
                    continue
                if d.get("status") != "completed":
                    continue
                rows.append(dict(board=board, scene=d["phase"], policy=pol, W=d["observed_mean_W"],
                                 dec_s=d["valid_detector_decisions_per_s"], calls=d["detector_calls"], positives=d["alerts"],
                                 source=os.path.relpath(f, DR), audit=d.get("reporting_status", "")))
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    with open(OUT_TSV, "w") as f:
        f.write("board\tscene\tpolicy\tmean_W\tdecisions_per_s\tdetector_calls\tpositives\tsource\taudit\n")
        for r in rows:
            f.write(f"{r['board']}\t{r['scene']}\t{r['policy']}\t{'rej.' if r['W'] is None else f'{r[chr(87)]:.2f}'}\t{r['dec_s']:.3f}\t{r['calls']}\t{r['positives']}\t{r['source']}\t{r['audit']}\n")
    by = {(r["board"], r["scene"], r["policy"]): r for r in rows}
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("% generated by train/edge_distill/make_camera_policy_table.py from results/camera_policy_rows.tsv\n")
        f.write("\\begin{table}[t]\\centering\\scriptsize\\caption{Whole-unit monitor-recapture power (W) and maximum decision rate: four 600\\,s "
                "policies per scene; power includes PSU and USB camera; models differ by board (the Pi~4s run the dense yolo26n); Orin maximum "
                "uses MAXN, other Orin windows 7\\,W; RPi~5 rip = earliest of six retained repeats; QAT-n = adapted yolo26n with NPU/CPU execution; rej. = window ran but its power record failed the "
                "sampling contract (gap $\\geq$5\\,s); -- = not run. A gated saving is only quoted where both comparators are accepted.}\\label{tab:camera}\n")
        f.write("\\begin{tabular}{l l r r r r r}\\hline\n board / deployed detector & scene & det.\\,1\\,Hz & gated & adaptive & max & max dec/s \\\\ \\hline\n")
        for b in ORDER:
            for scene in ("rip", "norip"):
                cells = []
                for p in POLICIES:
                    r = by.get((b, scene, p))
                    cells.append("--" if r is None else "rej." if r["W"] is None else f"{r['W']:.2f}")
                r = by.get((b, scene, "max_throughput"))
                if r is not None and r["W"] is None:
                    r = None
                if not any(by.get((b, scene, p)) is not None and by[(b, scene, p)]["W"] is not None for p in POLICIES):   # no accepted window yet
                    continue
                line = f" {LABEL[b] if scene == 'rip' else ''} & {'rip' if scene == 'rip' else 'no-rip'} & " + " & ".join(cells)
                line += f" & {r['dec_s']:.1f}" if r else " & --"
                f.write(line + " \\\\\n")
        f.write("\\hline\\end{tabular}\\end{table}\n")
    print(f"{len(rows)} rows -> {OUT_TSV}; table -> {OUT_TEX}")


if __name__ == "__main__":
    main()
