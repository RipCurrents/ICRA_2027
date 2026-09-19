"""LOCAL retrain of the RF-DETR-Seg-nano teacher on the 50 % LABELLED half of the s1.3
train videos (ICRA semi-supervised holdout; design: ICRA_2027/notes/design_semisup_holdout.md).

This is train_cluster/instance_segmentation/rfdetr/1_nano/rfdetr_seg_train.py (rfs-01, the
canonical teacher's recipe) with ONLY the following changed, all constants below:
  * TRAIN_JSON_PATH -> the filtered json (labelled videos only) that lives OUTSIDE the dataset tree;
  * local paths: dataset on the NVMe, no scratch staging, run/best/summary outputs under
    /mnt/linux/icra_edge/teacher_h50 (nothing is written into train_cluster/, models/ or labels/);
  * RUN_NAME suffix _h50.
Recipe: RFDETRSegNano, effective batch 64 (8 x 8 here instead of the cluster's 16 x 4 - the only deviation, memory), lr 1e-4, 60 epochs / early
stopping 10 on val ema segm mAP, 8 workers, eval_max_dets 50, fp16 eval, full val every epoch.
Env: conda rfdetr (rfdetr 1.6.5). Launch under systemd-run (see ICRA_2027/JOBS.md).
"""
import csv
import json
import os
import shutil

from rfdetr import RFDETRSegNano

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = "/mnt/linux/icra_edge/teacher_h50"           # bulky outputs (runs, best model, summary)
PROJECT = os.path.join(OUT_ROOT, "runs")

DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
INSTANCE_COCO_DIR = os.path.join(DATASET_ROOT, "labels", "instance_segmentation", "coco")
TRAIN_JSON = "train.json"  # (unused name kept for the summary) - the actual train json is:
TRAIN_JSON_PATH = "/home/user/RipBench/icra_edge_views/h50/train_h50_labelled.json"  # labelled half only
VAL_JSON_PATH = "/home/user/RipBench/icra_edge_views/h50/val_sub20.json"  # DEVIATION (2026-08-27 18:4x): every 20th val frame (518 imgs); was val_sub5 (15:45), originally the full val.json;
                      # = every 5th val frame (cheap per-epoch model selection; decided at the first checkpoint boundary)
VIEW_ROOT = "/home/user/RipBench/icra_edge_views/h50/rfdetr_seg_view"   # NVMe: jsons + image symlinks
STAGE_TO_SCRATCH = False
STAGE_TAR = os.path.join(DATASET_ROOT, "RipBench_v1.2.0_stage.tar")
SCRATCH_ROOT = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                            f"ripbench_stage_{os.environ.get('SLURM_JOB_ID', os.getpid())}")

RESUME_FROM = os.environ.get("RIPBENCH_RESUME_FROM") or None  # a lightning last.ckpt
EPOCHS = 20  # LOCAL DEVIATION (2026-08-27 19:5x): cluster recipe 60 / patience 10 (validated every epoch); here validation runs every
# 2nd epoch so patience 10 = 20 epochs without improvement -> up to 60 x 17 min = 17 h. The canonical teacher effectively trained
# 14 epochs (best 12); 20 keeps the holdout teacher comparable and the schedule (~01:30 end). Original comment: rfdetr-specific budget, deviation documented: measured epoch
# cost on L40 is ~5h (nano) to ~12h (2xlarge) - the matrix-wide 300/50 is
# unreachable (months/variant). Convergence is fast (rip AP50 0.65 by epoch
# 4), so the cap mainly bounds the early-stopping tail
BATCH_SIZE = 8   # LOCAL DEVIATION (2026-08-27): 16 x 4 on the cluster's 48 GB L40; on the shared 24 GB 4090 batch 16
GRAD_ACCUM_STEPS = 8  # peaked at 20.2 GB and OOM'd at 13:22 -> 8 x 8 keeps the EFFECTIVE batch 64 (same optimisation,
# LayerNorm model, no batch statistics) at ~half the activation memory; documented in ICRA_2027/JOBS.md
LR = 1e-4
EARLY_STOPPING_PATIENCE = 10  # see EPOCHS note; on ema segm mAP
NUM_WORKERS = 8  # mask-carrying batches: keep modest (host-RAM cgroup lesson)
EVAL_MAX_DETS = 10  # DEVIATION (2026-08-27 15:45): 50 on the cluster; <= a handful of rips per image, 10 is still generous
EVAL_INTERVAL = 2  # DEVIATION (2026-08-27 16:4x): validate every 2nd epoch (default 1) - the per-epoch validation is CPU-bound
                   # (~45 min even on the 2k-frame subset); with per-epoch checkpoints + the post-hoc full-val re-ranking the
                   # per-epoch metric only drives early stopping (patience counts validation events)
CHECKPOINT_INTERVAL = 1  # DEVIATION (2026-08-27, critic settlement): keep EVERY epoch's lightning checkpoint (default 10) so the
                         # final teacher epoch is chosen by the ORIGINAL metric (full val, 50 dets, benchmark tools) after
                         # training - the cheap per-epoch proxy then only drives early stopping (rank_teacher_epochs.py)
# LOCAL NOTE (2026-08-27): the per-epoch full-val evaluation is CPU-bound (single-threaded RLE encoding of up to
# EVAL_MAX_DETS 4K masks per image) and took >1 h/epoch on this box; if that holds, restart from last.ckpt with
# EVAL_MAX_DETS = 10 (val-metric/model-selection cost only; training unchanged) - decided at the first checkpoint boundary.
FP16_EVAL = True  # faster per-epoch val (v1.1.7 cluster setting)
NAME = "RipBench_v1.2.0_rfdetr_seg_nano_h50"
RUN_NAME = NAME + (f"_j{os.environ['SLURM_JOB_ID']}" if "SLURM_JOB_ID" in os.environ else "")


def stage_to_scratch():
    """Extract the pre-built stage tar (one sequential read) to node-local
    NVMe - per-file copying off /data is orders of magnitude slower. The tar
    is built and shipped from the local PC by tools/sync_dataset_to_cluster.sh."""
    import subprocess
    import time
    assert os.path.isfile(STAGE_TAR), (
        f"stage tar not found: {STAGE_TAR}\n"
        "build+ship it from the local PC: bash tools/sync_dataset_to_cluster.sh"
    )
    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    print(f"extracting {STAGE_TAR} ({os.path.getsize(STAGE_TAR) / 1e9:.0f} GB)...", flush=True)
    t0 = time.time()
    subprocess.run(["tar", "-xf", STAGE_TAR, "-C", SCRATCH_ROOT,
                    "--checkpoint=100000", "--checkpoint-action=echo=%u records done"],
                   check=True)
    print(f"extracted in {time.time() - t0:.0f}s", flush=True)
    return os.path.join(SCRATCH_ROOT, "images")


def build_view(images_root):
    if os.path.exists(VIEW_ROOT):
        shutil.rmtree(VIEW_ROOT)
    splits = {"train": TRAIN_JSON, "valid": "val.json", "test": "test.json"}
    for split_dir, json_name in splits.items():
        d = os.path.join(VIEW_ROOT, split_dir)
        os.makedirs(d)
        src_json = TRAIN_JSON_PATH if split_dir == "train" else (VAL_JSON_PATH if (split_dir == "valid" and VAL_JSON_PATH) else
                                                                 os.path.join(INSTANCE_COCO_DIR, json_name))
        shutil.copy2(src_json, os.path.join(d, "_annotations.coco.json"))
        for source in ("sampled_frames", "additional_data"):
            src = os.path.join(images_root, source)
            if os.path.isdir(src):
                os.symlink(src, os.path.join(d, source))
    return VIEW_ROOT


def print_summary(run_dir, best_dst):
    """Cluster-convention summary row: best/stopped epoch + the val metric
    trajectory from lightning's metrics.csv (rfdetr writes no separate test
    row there; benchmark-grade test numbers come from the eval/ pipeline)."""
    from datetime import datetime

    best_epoch = stopped_epoch = None
    best_val = {}
    metric_keys = ("val/ema_segm_mAP_50", "val/ema_segm_mAP_50_95",
                   "val/segm_mAP_50", "val/segm_mAP_50_95",
                   "val/ema_mAP_50_95", "val/mAP_50_95")
    select_on = "val/ema_segm_mAP_50_95"
    best_metric = None
    csv_path = os.path.join(run_dir, "metrics.csv")
    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if not row.get("epoch"):
                    continue
                epoch = int(float(row["epoch"]))
                stopped_epoch = max(stopped_epoch or 0, epoch + 1)
                v = row.get(select_on) or row.get("val/segm_mAP_50_95")
                if v:
                    v = float(v)
                    if best_metric is None or v > best_metric:
                        best_metric, best_epoch = v, epoch + 1
                        best_val = {k: float(row[k]) for k in metric_keys if row.get(k)}

    job = os.environ.get("SLURM_JOB_NAME", "local")
    lines = ["=" * 70, f"RUN SUMMARY  {RUN_NAME}  (job: {job})",
             f"best epoch: {best_epoch} (selected on {select_on}) | "
             f"stopped after epoch: {stopped_epoch} / {EPOCHS} max",
             "best-epoch val metrics: " + json.dumps(best_val, default=float),
             f"best model copy: {best_dst}", "=" * 70]
    print("\n".join(lines), flush=True)

    header = ("timestamp\tjob\trun_name\tbest_epoch\tstopped_epoch\tmax_epochs"
              "\tval_metrics\ttest_metrics\tbest_model\n")
    row = "\t".join([datetime.now().strftime("%Y-%m-%d %H:%M"), job, RUN_NAME,
                     str(best_epoch), str(stopped_epoch), str(EPOCHS),
                     json.dumps(best_val, default=float),
                     json.dumps({"note": "test via eval/ pipeline"}), str(best_dst)])
    for tsv in (os.path.join(OUT_ROOT, "results_summary.tsv"),):
        write_header = not os.path.isfile(tsv)
        with open(tsv, "a") as f:
            if write_header:
                f.write(header)
            f.write(row + "\n")
        print(f"summary line appended to {tsv}", flush=True)


def collect_best(run_dir):
    import re as _re
    src = None
    for cand in ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "last.ckpt"):
        p = os.path.join(run_dir, cand)
        if os.path.isfile(p):
            src = p
            break
    if src is None:
        print("WARNING: no checkpoint found to collect", flush=True)
        return None
    stem = "1_seg_rfdetr_nano_h50"
    best_dir = os.path.join(OUT_ROOT, "best_models")
    os.makedirs(best_dir, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    dst = os.path.join(best_dir, f"{stem}_best_{os.path.basename(src).replace('checkpoint_', '').replace('.pth', '').replace('.ckpt', '')}_j{job_id}.pth")
    shutil.copy2(src, dst)
    return dst


def main():
    assert os.path.isdir(DATASET_ROOT), f"DATASET_ROOT not found: {DATASET_ROOT}"
    images_root = stage_to_scratch() if STAGE_TO_SCRATCH else os.path.join(DATASET_ROOT, "images")
    dataset_dir = build_view(images_root)
    print(f"Prepared rfdetr seg view at {dataset_dir} (train json: {TRAIN_JSON_PATH})")

    run_dir = os.path.join(PROJECT, RUN_NAME)
    model = RFDETRSegNano()
    kwargs = dict(dataset_dir=dataset_dir, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  grad_accum_steps=GRAD_ACCUM_STEPS, lr=LR,
                  early_stopping=True, early_stopping_patience=EARLY_STOPPING_PATIENCE,
                  num_workers=NUM_WORKERS, run_test=True,
                  eval_max_dets=EVAL_MAX_DETS, fp16_eval=FP16_EVAL,
                  checkpoint_interval=CHECKPOINT_INTERVAL, eval_interval=EVAL_INTERVAL,
                  output_dir=run_dir, progress_bar=False)  # progress bar spams the .out
    if RESUME_FROM:
        kwargs["resume"] = RESUME_FROM
    model.train(**kwargs)

    best_dst = collect_best(run_dir)
    print_summary(run_dir, best_dst)


if __name__ == "__main__":
    main()
