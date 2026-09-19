#!/usr/bin/env python3
"""Student training + benchmark-grade evaluation queue (ICRA edge distillation).

Trains yolo26 students on a VIEW (canonical expert labels, or a teacher pseudo-
label view from build_teacher_view.py) on the local 4090, one run at a time,
then evaluates each best.pt with the benchmark pipeline (eval/predict_ultralytics
+ eval/evaluate_detection, imported unmodified) against BOTH box GT views:
  * bbox_from_instance (matching the training convention)  -> results/student_eval_bfi.tsv
  * manual tight bbox (the canonical benchmark GT)         -> results/student_eval_tight.tsv
Never touches eval/results/. Predictions -> predictions/students/ (on /mnt/linux via
symlink), runs -> runs/ (idem), best.pt copies -> best_models/.

Recipe = the benchmark yolo26 recipe (COCO-pretrained yolo26n.pt init, imgsz 640,
SGD-auto, AMP) with the batch size fixed to BATCH (the cluster's AutoBatch picked
per-node batches - a fixed batch keeps every local variant comparable). Epoch
budgets: canonical views 300 ep / patience 50 (benchmark); dense views scale the
per-epoch data 6x, so EPOCHS_DENSE / PATIENCE_DENSE (both early-stop on val
fitness = ultralytics mAP50-95, as on the cluster). Native ultralytics feature KD
= extra kwargs distill_model=<same-task yolo26x ckpt>, dis=6.0.

Idempotent / resumable like train/multiseed_local: finished runs are skipped
(finished-checkpoint guard), unfinished ones resume from last.pt, eval rows
dedupe by model+task+split. Refuses to start a training while another process
holds > GPU_BUSY_MB on the GPU (shared 4090 policy). No CLI args - edit QUEUE.
Env: conda yoloV26 (ultralytics 8.4.92).
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_ROOT = "/home/user/RipBench/RipBench_v1.2.0"
VIEWS_ROOT = "/home/user/RipBench/icra_edge_views"
MODELS_ROOT = os.path.join(DATASET_ROOT, "models")

RUNS_DIR = os.path.join(HERE, "runs")
BEST_DIR = os.path.join(HERE, "best_models")
PRED_DIR = os.path.join(HERE, "predictions", "students")
RESULTS_DIR = os.path.join(HERE, "results")
TRAIN_TSV = os.path.join(RESULTS_DIR, "student_train_summary.tsv")
EVAL_TSV = {"bbox_from_instance": os.path.join(RESULTS_DIR, "student_eval_bfi.tsv"),
            "bbox": os.path.join(RESULTS_DIR, "student_eval_tight.tsv")}

INIT_DET = os.path.join(REPO_ROOT, "yolo26n.pt")  # COCO-pretrained, same init as the benchmark
KD_TEACHER_BFI_X = os.path.join(MODELS_ROOT, "bbox_from_instance", "best_models",
                                "9_bfi_yolo26_xlarge_best_e21_j2853807.pt")
IMGSZ, BATCH, WORKERS = 640, 32, 6
EPOCHS_CANON, PATIENCE_CANON = 300, 50
EPOCHS_DENSE, PATIENCE_DENSE = 60, 15
GPU_BUSY_MB = 10_000
EVAL_TASKS = ("bbox_from_instance", "bbox")

# (run name, view dir name under VIEWS_ROOT, init weights, seed, epochs, patience, extra train kwargs)
INIT_S = "/mnt/linux/RipBench/train/ripvis_transfer_ablation/yolo26s.pt"  # COCO-pretrained yolo26s (edge-ladder student)
QUEUE = [  # local wave 2 (2026-08-27): third seeds of the yolo26s edge-ladder rows (seeds 1-2 on the cluster)
    ("canon_bfi_s_s3", "canonical_bfi", INIT_S, 3, EPOCHS_CANON, PATIENCE_CANON, {}),
    ("t1_nano_k025_t024_s1_s_s3", "t1_rfdetr_seg_nano_k0.25_t0.24_s1", INIT_S, 3, EPOCHS_DENSE, PATIENCE_DENSE, {}),
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def gpu_other_usage_mb():
    """MB held by processes other than ours (nvidia-smi query)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return 0
    total = 0
    for line in out.strip().splitlines():
        pid, mem = [x.strip() for x in line.split(",")]
        if int(pid) != os.getpid():
            total += int(mem)
    return total


def wait_for_gpu():
    while (u := gpu_other_usage_mb()) > GPU_BUSY_MB:
        print(f"[gpu] another process holds {u} MB > {GPU_BUSY_MB} - waiting 10 min", flush=True)
        time.sleep(600)


def ensure_canonical_view(name, task):
    """Plain expert-label view (two symlinks + copied manifests), like multiseed_views."""
    view = os.path.join(VIEWS_ROOT, name)
    yaml_path = os.path.join(view, "data.yaml")
    if os.path.isfile(yaml_path):
        return yaml_path
    os.makedirs(view, exist_ok=True)
    manifest = os.path.join(DATASET_ROOT, "labels", task, "yolo26")
    if not os.path.lexists(os.path.join(view, "images")):
        os.symlink(os.path.join(DATASET_ROOT, "images"), os.path.join(view, "images"))
    # labels/: REAL dirs down to the class level, symlinks only at the leaves - ultralytics
    # writes its scan .cache next to the first label's parent dir, which must land in the
    # view, never inside the shared dataset label tree
    for src in ("sampled_frames", "additional_data"):
        for cls in ("rips", "no-rips"):
            target = os.path.join(manifest, "labels", src, cls)
            if os.path.isdir(target):
                d = os.path.join(view, "labels", src)
                os.makedirs(d, exist_ok=True)
                if not os.path.lexists(os.path.join(d, cls)):
                    os.symlink(target, os.path.join(d, cls))
    for split in ("train", "val", "test"):
        shutil.copy2(os.path.join(manifest, f"{split}.txt"), os.path.join(view, f"{split}.txt"))
    with open(yaml_path, "w") as f:
        f.write(f"path: {view}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: rip_current\n")
    return yaml_path


def data_yaml_for(view_name):
    if view_name == "canonical_bfi":
        return ensure_canonical_view(view_name, "bbox_from_instance")
    if view_name == "canonical_tight":
        return ensure_canonical_view(view_name, "bbox")
    y = os.path.join(VIEWS_ROOT, view_name, "data.yaml")
    assert os.path.isfile(y), f"view not built: {y} (run build_teacher_view.py)"
    return y


def training_finished(ckpt_path):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("optimizer") is None or ckpt.get("epoch", 0) == -1


def summary_has(name):
    return os.path.isfile(TRAIN_TSV) and any(l.split("\t")[1:2] == [name] for l in open(TRAIN_TSV))


def tsv_has(tsv, model, split):
    if not os.path.isfile(tsv):
        return False
    for line in open(tsv):
        p = line.rstrip("\n").split("\t")
        if len(p) > 4 and p[1] == model and p[4] == split:
            return True
    return False


def train_one(name, view_name, init, seed, epochs, patience, extra):
    from ultralytics import YOLO
    run_dir = os.path.join(RUNS_DIR, name)
    last = os.path.join(run_dir, "weights", "last.pt")
    best = os.path.join(run_dir, "weights", "best.pt")
    data_yaml = data_yaml_for(view_name)
    existing = sorted(f for f in os.listdir(BEST_DIR) if f.startswith(name + "_best_e")) if os.path.isdir(BEST_DIR) else []
    if summary_has(name) and existing:
        print(f"[{name}] already trained + summarized -> {existing[-1]}")
        return os.path.join(BEST_DIR, existing[-1])
    finished = os.path.isfile(last) and training_finished(last)
    resume = os.path.isfile(last) and not finished
    if finished:
        model = YOLO(best if os.path.isfile(best) else last)
    else:
        wait_for_gpu()
        model = YOLO(last if resume else init)
        kwargs = dict(data=data_yaml, epochs=epochs, patience=patience, batch=BATCH, imgsz=IMGSZ,
                      workers=WORKERS, seed=seed, project=RUNS_DIR, name=name, exist_ok=True,
                      resume=resume, **extra)
        print(f"[{name}] training {init} on {view_name} seed={seed} epochs={epochs} patience={patience} "
              f"extra={extra} resume={resume}", flush=True)
        model.train(**kwargs)
    trainer = getattr(model, "trainer", None)
    best_epoch = stopped = None
    if trainer is not None:
        stopped = getattr(trainer, "epoch", None)
        stopped = stopped + 1 if stopped is not None else None
        best_epoch = getattr(getattr(trainer, "stopper", None), "best_epoch", None)
    if best_epoch is None:  # completed earlier (resume of a finished run): recover from results.csv
        import csv
        csv_path = os.path.join(run_dir, "results.csv")
        if os.path.isfile(csv_path):
            rows = [{k.strip(): v.strip() for k, v in r.items() if k} for r in csv.DictReader(open(csv_path))]
            rows = [r for r in rows if r.get("epoch")]
            if rows:
                stopped = int(float(rows[-1]["epoch"]))
                best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", "nan") or "nan"))
                best_epoch = int(float(best_row["epoch"]))
    os.makedirs(BEST_DIR, exist_ok=True)
    dst = os.path.join(BEST_DIR, f"{name}_best_e{best_epoch}.pt")
    if os.path.isfile(best):
        shutil.copy2(best, dst)
    n_train = sum(1 for l in open(os.path.join(os.path.dirname(data_yaml), "train.txt")) if l.strip())
    header = ("timestamp\tname\tview\tinit\tseed\tepochs_max\tpatience\tbest_epoch\tstopped_epoch"
              "\tn_train_images\tsamples_seen\textra\tbest_model\n")
    row = "\t".join([datetime.now().strftime("%Y-%m-%d %H:%M"), name, view_name, os.path.basename(init),
                     str(seed), str(epochs), str(patience), str(best_epoch), str(stopped), str(n_train),
                     str(n_train * stopped if stopped else ""), json.dumps(extra, default=str), dst])
    write_header = not os.path.isfile(TRAIN_TSV)
    with open(TRAIN_TSV, "a") as f:
        if write_header:
            f.write(header)
        f.write(row + "\n")
    print(f"[{name}] done: best epoch {best_epoch}, stopped {stopped}, {n_train} train images/epoch", flush=True)
    return dst if os.path.isfile(dst) else best


def bench_eval(ckpt):
    """Benchmark-grade eval on both box GT views (imported eval/ tooling, own TSVs)."""
    PU = load_module("pu_edge", os.path.join(REPO_ROOT, "eval", "predict_ultralytics.py"))
    ED = load_module("ed_edge", os.path.join(REPO_ROOT, "eval", "evaluate_detection.py"))
    PU.OUT_DIR, PU.MEASURE_FPS, PU.SKIP_EXISTING, PU.SPLITS = PRED_DIR, False, True, ("val", "test")
    ED.SAVE_SWEEP = False
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    os.makedirs(PRED_DIR, exist_ok=True)
    PU.process_checkpoint(ckpt)
    for task in EVAL_TASKS:
        if tsv_has(EVAL_TSV[task], stem, "test"):
            print(f"[{stem}] already evaluated on {task}")
            continue
        ED.TASK, ED.MODE, ED.OUT_TSV = task, "bbox", EVAL_TSV[task]
        base = os.path.join(DATASET_ROOT, "labels", task, "coco")
        ED.GT_VAL, ED.GT_TEST = os.path.join(base, "val.json"), os.path.join(base, "test.json")
        gt_v, gt_t = ED.COCO(ED.GT_VAL), ED.COCO(ED.GT_TEST)
        ED.evaluate_pair(gt_v, gt_t, os.path.join(PRED_DIR, f"{stem}_preds_val.json"),
                         os.path.join(PRED_DIR, f"{stem}_preds_test.json"), stem)


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    failures = []
    for name, view, init, seed, epochs, patience, extra in QUEUE:
        try:
            ckpt = train_one(name, view, init, seed, epochs, patience, extra)
            if ckpt and os.path.isfile(ckpt):
                bench_eval(ckpt)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"QUEUE LEG FAILED {name}: {e!r}", flush=True)
            failures.append(name)
    if failures:
        raise SystemExit(f"queue finished with failures: {failures}")
    print("STUDENT QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
