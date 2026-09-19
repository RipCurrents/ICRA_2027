> Paths in this ledger are relative to the working folder it was written in: `results/` is `results/`, `manifests/` is `data/`, scripts are in `code/`, `ICRA_2027/device_results/` files are in `results/device/` when the paper cites them and otherwise in the authors' working archive, `plans/TEST_CONSUMPTION.md` is `paper/TEST_CONSUMPTION.md`, and bulky inputs under `/mnt/linux/icra_edge/` (models, checkpoints, label views, device dumps) are not part of this repository.

# train/edge_distill — RESULTS (ICRA 2027 edge distillation)

All numbers on split **s1.3**, benchmark evaluator `eval/evaluate_detection.py` (conf swept on
val by F2@IoU.50, frozen, applied to test). Every row names its provenance file. Nothing here is
appended to `eval/results/` (benchmark TSVs stay untouched).

## Teachers, box-ified (test) — provenance `results/teacher_boxified_eval.tsv` (2026-08-25)
Boxes = mask bounding rects of the benchmark RF-DETR-Seg RLE dumps
(`RipBench_v1.2.0/models/instance_segmentation/best_models/predictions/`), scored as plain boxes.

| teacher | GT view | conf (val) | F2@50 | P | R | AP50 | AP75 | AP50-95 |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-Seg-nano (canonical) | bbox_from_instance | 0.24 | 0.697 | 0.717 | 0.692 | 0.702 | 0.261 | 0.328 |
| RF-DETR-Seg-nano +additional | bbox_from_instance | 0.24 | 0.722 | 0.646 | 0.743 | 0.741 | 0.277 | 0.349 |
| RF-DETR-Seg-small (canonical) | bbox_from_instance | 0.26 | 0.672 | 0.680 | 0.671 | 0.661 | 0.193 | 0.297 |
| RF-DETR-Seg-small +additional | bbox_from_instance | 0.24 | 0.726 | 0.659 | 0.745 | 0.740 | 0.322 | 0.372 |
| RF-DETR-Seg-nano (canonical) | bbox (manual tight) | 0.26 | 0.688 | 0.730 | 0.678 | 0.688 | 0.193 | 0.297 |
| RF-DETR-Seg-nano +additional | bbox (manual tight) | 0.24 | 0.702 | 0.629 | 0.723 | 0.713 | 0.260 | 0.331 |
| RF-DETR-Seg-small (canonical) | bbox (manual tight) | 0.26 | 0.664 | 0.671 | 0.662 | 0.643 | 0.206 | 0.287 |

Reference students (benchmark rows, cluster seed 0, `eval/results/bbox_from_instance_eval.tsv` /
`bbox_eval.tsv`, read-only): yolo26n bfi F2 0.437 / AP50 0.352 (+additional 0.533 / 0.496);
yolo26n tight F2 0.444 / AP50 0.409 (+additional 0.538 / 0.492).

### Heavy reference row for the Pareto plot (added 2026-08-27; camera-ready session's numbers, READ-ONLY sources)
RF-DETR DETECTION nano (`models/bbox/best_models/1_det_rfdetr_nano_best_best_ema_j3178618.pth`, rfdetr 1.6.5, 384 px):
test F2 0.699 / AP50-95 0.106 at the val-frozen conf 0.23 on the manual tight box GT (`eval/results/bbox_eval.tsv`,
2026-08-27 09:25); 73.2 fps on the 4090 (`eval/results/fps_operating.tsv`: batch 1, FP32, 200 timed frames, measured
AT the operating conf, 11.3 ms inference + 2.4 ms preprocess). It is the strongest light single-frame detector on the
benchmark and goes on the plot as the heavy reference; ONNX export: `export_rfdetr_det.py` → `/mnt/linux/icra_edge/exports/rfdetr_det/<stem>/inference_model.onnx` (384 px,
30.15 M params, 114 MB, onnxsim; `results/export_manifest_rfdetr.tsv`); validation against the benchmark row with the
unmodified evaluator: `rfdetr_onnx_bench.py` → `results/reference_eval_tight.tsv`: **ONNX path (onnxruntime CPU, own
numpy pre/post) test F2 0.706 / AP50 0.710 / AP50-95 0.249 at val-frozen conf 0.26** vs the camera-ready PyTorch row
0.699 / 0.699 / 0.244 at conf 0.23 — reproduced within +0.007 F2 / +0.011 AP50 (cv2 vs PIL resize and the full 300-query
dump explain the small excess; the benchmark row stays the canonical accuracy, the ONNX row is the device reference). Decode facts verified against
the PyTorch forward (ONNX outputs identical): inputs square-resized to 384 with ImageNet mean/std (RGB); outputs `dets`
(1,300,4) cx,cy,w,h normalised and `labels` (1,300,2) logits where the rip class is index **0** (position in
`class_names`, not the COCO id — a first pass with index 1 scored F2 0.01). Note the GT-view rule: its F2 is on the tight view; compare with the
students' `student_eval_tight.tsv` rows, not the bfi ones.

## Views built (train split; loader-verified through ultralytics YOLODataset)
| view | GT frames (expert boxes) | dense frames (boxes) | total train images | provenance |
|---|---|---|---|---|
| interp_gt_s1 (teacher-free control) | 31,083 (29,400) | 160,530 (155,176 interpolated; 54,445 no-rip negatives) | 191,613 | `views/interp_gt_s1/build_info.json` |
| t1_rfdetr_seg_nano_k0.25_t0.24_s1 (headline) | 31,083 (29,400) | 160,175 (155,241 teacher boxes on 105,730 rip-video frames = 99.7% of un-annotated rip frames kept, 1.47 boxes/frame; 54,445 no-rip negatives) | 191,258 | `views/t1_rfdetr_seg_nano_k0.25_t0.24_s1/build_info.json`, log `/mnt/linux/icra_edge/build_teacher_view_k025_t024_s1.log` |
| t1_rfdetr_seg_nano_k0.25_t0.15_s1 | 31,083 (29,400) | 160,485 (158,139 teacher boxes) | 191,568 | `views/t1_rfdetr_seg_nano_k0.25_t0.15_s1/build_info.json` |
| t1_rfdetr_seg_nano_k0.25_t0.24_s1_keepempty | 31,083 (29,400) | 160,536 (155,241 teacher boxes) | 191,619 | `views/t1_rfdetr_seg_nano_k0.25_t0.24_s1_keepempty/build_info.json` |
| t1_rfdetr_seg_nano_k0.25_t0.24_s2 | 31,083 (29,400) | 77,304 (75,501 teacher boxes) | 108,387 | `views/t1_rfdetr_seg_nano_k0.25_t0.24_s2/build_info.json` |
| t1_rfdetr_seg_nano_k0.25_t0.24_s4 | 31,083 (29,400) | 38,526 (37,697 teacher boxes) | 69,609 | `views/t1_rfdetr_seg_nano_k0.25_t0.24_s4/build_info.json` |
| t1_rfdetr_seg_nano_k0.25_t0.35_s1 | 31,083 (29,400) | 159,221 (152,369 teacher boxes) | 190,304 | `views/t1_rfdetr_seg_nano_k0.25_t0.35_s1/build_info.json` |
| t1_rfdetr_seg_nano_k0_t0.24_s1 | 31,083 (29,400) | 160,533 (157,251 teacher boxes) | 191,616 | `views/t1_rfdetr_seg_nano_k0_t0.24_s1/build_info.json` |

| t1_rfdetr_seg_nano_add_k0.25_t0.24_s1 (+additional teacher) | 31,083 (29,400) | 160,134 (155,562 teacher boxes) | 191,217 | `views/t1_rfdetr_seg_nano_add_k0.25_t0.24_s1/build_info.json` |
| t1_rfdetr_seg_nano_k0.25_t0.24_s1_noneg (no dense negatives) | 31,083 (29,400) | 105,730 (155,241 teacher boxes; no no-rip dense frames) | 136,813 | `views/t1_rfdetr_seg_nano_k0.25_t0.24_s1_noneg/build_info.json` |
Native frames: 191,619 extracted from the 165 train videos (`manifests/train_frames_manifest.tsv`,
165/165 pixel checks OK).

## Latency — 4090 stand-in ladder (until the Jetson arrives; `results/latency_4090.tsv`, `latency_ladder_4090.py`)
onnxruntime CUDA EP, batch 1, 200 warm-up + 1000 timed test frames × 5 repeats, GPU otherwise idle, 2026-08-26.
| model | input | FP32 infer ms | FP16 infer ms | pre-process ms (CPU letterbox) |
|---|---|---|---|---|
| yolo26n (canonical or dense — same architecture) | 640 | 2.20 ± 0.01 | 2.15 ± 0.12 | 2.3–2.5 |
| | 512 | 1.92 ± 0.02 | 1.84 ± 0.01 | 1.3–1.4 |
| | 384 | 1.69 ± 0.01 | 1.66 ± 0.00 | 0.8–0.9 |
On a 4090 a nano is launch-bound (FP16 −2…−5 %); the resolution ladder matters on the device, not here.
Params 2.50 M; GFLOPs 5.8 / 3.7 / 2.1 at 640 / 512 / 384 (`results/export_manifest.tsv`).
Board path validated (`board_bench.py`, ONNX + numpy decode/NMS, CPU, full test split): dense yolo26n @384 → F2 0.493 /
AP50 0.458 / AP50-95 0.224 vs the PyTorch predictor's 0.498 / 0.454 / 0.211 at the same val-frozen conf (0.04) — board rows
are comparable to the 4090 rows within ±0.005 F2 (dump: `/mnt/linux/icra_edge/board_runs/`).

### Accuracy vs input resolution (test, mean ± std over 3 seeds for every row; `eval_resolution_ladder.py`, rows `_img512/_img384` in the eval TSVs; updated 2026-08-27 13:30)
| student | input | GFLOPs | bfi F2@50 | bfi AP50 | bfi AP50-95 | tight F2@50 | tight AP50 |
|---|---|---|---|---|---|---|---|
| canonical yolo26n | 640 | 5.8 | 0.483 ± 0.022 | 0.444 ± 0.031 | 0.191 ± 0.019 | 0.500 ± 0.026 | 0.465 ± 0.036 |
| | 512 | 3.7 | 0.471 ± 0.015 | 0.442 ± 0.013 | 0.186 ± 0.013 | 0.495 ± 0.015 | 0.463 ± 0.013 |
| | 384 | 2.1 | 0.484 ± 0.007 | 0.428 ± 0.018 | 0.185 ± 0.014 | 0.520 ± 0.005 | 0.477 ± 0.011 |
| dense (T1) yolo26n | 640 | 5.8 | 0.531 ± 0.016 | 0.505 ± 0.026 | 0.232 ± 0.013 | 0.532 ± 0.008 | 0.504 ± 0.020 |
| | 512 | 3.7 | 0.524 ± 0.022 | 0.504 ± 0.026 | 0.234 ± 0.010 | 0.525 ± 0.025 | 0.495 ± 0.028 |
| | 384 | 2.1 | 0.527 ± 0.023 | 0.496 ± 0.032 | 0.222 ± 0.020 | 0.530 ± 0.019 | 0.491 ± 0.028 |
| canonical yolo26s (9.95 M params) | 640 | 22.5 | 0.479 ± 0.054 | 0.435 ± 0.067 | 0.166 ± 0.012 | 0.498 ± 0.035 | 0.480 ± 0.047 |
| | 512 | 14.4 | 0.485 ± 0.046 | 0.428 ± 0.033 | 0.168 ± 0.014 | 0.503 ± 0.026 | 0.465 ± 0.011 |
| | 384 | 8.1 | 0.482 ± 0.029 | 0.448 ± 0.039 | 0.171 ± 0.004 | 0.495 ± 0.007 | 0.467 ± 0.016 |
| dense (T1) yolo26s | 640 | 22.5 | 0.550 ± 0.013 | 0.505 ± 0.020 | 0.226 ± 0.015 | 0.557 ± 0.018 | 0.523 ± 0.022 |
| | 512 | 14.4 | 0.553 ± 0.022 | 0.529 ± 0.032 | 0.226 ± 0.011 | 0.555 ± 0.015 | 0.538 ± 0.015 |
| | 384 | 8.1 | **0.578 ± 0.010** | **0.558 ± 0.012** | 0.245 ± 0.013 | 0.566 ± 0.008 | 0.548 ± 0.009 |
Reading (all rows 3 seeds; conf re-selected on val per resolution as the evaluator prescribes; GFLOPs from
`results/export_manifest.tsv`): yolo26n is FLAT from 640 to 384 px (±0.01 F2) at 2.8× fewer GFLOPs. SURPRISE, replicated on
3 seeds (std 0.010) and flagged as such: the dense yolo26s is BEST at 384 px (0.578 vs 0.550 at 640; AP50 +0.05, AP50-95 +0.02)
while the canonical yolo26s is flat. The small-object-false-positive hypothesis was TESTED and REFUTED (2026-08-27): FPs with
area < 1 % of the image are 15–19 % of all FPs at both resolutions; the 384-px gain is a RECALL gain at similar precision
(TP 3626→3830 / 3403→3591 / 3585→3651 across seeds, FP counts similar). Mechanism open (scale match between the 1280-px
dense training frames downsampled to 640 and the 4K test frames downsampled to 384 is the next candidate); reported as an
empirical finding. Consequences for the edge ladder: dense yolo26s @384 (8.1 GFLOPs, 0.578) beats canonical yolo26s @640 (22.5 GFLOPs,
0.479) by +0.10 F2; dense yolo26n @384 (2.1 GFLOPs, 0.527) matches canonical yolo26s @640.

## Cls-gated cascade — simulation on the benchmark dumps (`cascade_simulation.py` → `results/cascade_sim.tsv`, 2026-08-27)
Gate = benchmark yolo26n-cls 224 (1.53 M params, frame F2 0.870); detector = dense yolo26n s1 (val-frozen conf 0.04).
Gate threshold selected on val (largest thr with ≤ 0.005 F2 loss) = 0.50; k-of-n temporal confirmation changes nothing
(the gate's decisions are already temporally consistent).
| detector | gate thr | test F2@50 (always-on) | detector runs on | compute ratio (GFLOPs stand-in) |
|---|---|---|---|---|
| dense yolo26n s1 | 0.50 (val) | 0.511 (0.513) | 47.0 % of frames | 0.57 |
| dense yolo26n s1 | 0.10 | 0.514 (0.513) | 53.2 % | 0.64 |
| canonical yolo26n s1 | 0.50 | 0.484 (0.497) | 47.0 % | 0.57 |
Gate pass rates on test (thr 0.50): rip frames 84.5 %, no-rip frames 14.9 %. The test split is 46 % rip frames by
construction, so the 47 % duty is a pessimistic bound: at a realistic 5 % / 20 % rip prevalence the detector would run on
18 % / 29 % of frames (gate FPR-dominated). Energy numbers replace the GFLOPs stand-in once measured on the device.

## Students (test; provenance `results/student_eval_bfi.tsv` / `student_eval_tight.tsv` / `student_train_summary.tsv`)
Recipe: yolo26n COCO init, imgsz 640, **fixed batch 32**, workers 6/16, SGD-auto, AMP; canonical 300 ep / patience 50,
dense views 60 / 15; early stop on val fitness (mAP50-95). CIs = video-level bootstrap (n=71 test videos, B=1000, seed 0,
`bootstrap_cis.py` -> `results/student_bootstrap_cis.tsv`), stated once per row.
NOTE (recipe, verified 2026-08-26): under this recipe the canonical yolo26n baseline lands ~0.06 F2 ABOVE the benchmark
seed-0 nano rows (which used AutoBatch = batch 73 on an L40, per the bfi-01/det-01 job logs in the mirror; batch 32 = ~2.3x more optimizer steps per epoch) — all paper comparisons use the rows below, never the benchmark rows.

| run | view | seed | best/stopped ep | bfi F2@50 [CI] | bfi AP50 | bfi AP50-95 | tight F2@50 | tight AP50 | where |
|---|---|---|---|---|---|---|---|---|---|
| canon_bfi_n_s1 | canonical expert labels | 1 | 18 / 68 | 0.497 [0.261, 0.697] | 0.469 | 0.210 | 0.520 | 0.497 | local 4090 |
| canon_bfi_n_s2 | canonical expert labels | 2 | 8 / 58 | 0.501 [0.258, 0.660] | 0.463 | 0.199 | 0.518 | 0.483 | the cluster 3172372 (local recompute identical) |
| canon_bfi_n_s3 | canonical expert labels | 3 | 22 / 72 | 0.453 [0.181, 0.678] | 0.400 | 0.165 | 0.463 | 0.414 | the cluster 3172376 |
| t1_nano_k025_t024_s1_n_s1 (HEADLINE view) | T1 teacher view, full density (191,258 imgs) | 1 | 2 / 17 | 0.513 [0.263, 0.703] | 0.477 | 0.216 | 0.527 | 0.481 | local 4090 (3.25M samples seen) |
| t1_nano_k025_t024_s1_n_s2 (HEADLINE view) | T1 teacher view, full density | 2 | 2 / 17 | 0.527 [0.265, 0.695] | 0.499 | 0.234 | 0.526 | 0.501 | the cluster 3172377 |
| t1_nano_k025_t024_s1_n_s3 (HEADLINE view) | T1 teacher view, full density | 3 | 2 / 17 | 0.552 [0.287, 0.724] | 0.539 | 0.247 | 0.544 | 0.530 | the cluster 3172378 |
| interp_gt_s1_n_s1 (CONTROL, teacher-free) | interpolated expert masks → boxes (191,613 imgs) | 1 | 2 / 17 | 0.471 [0.192, 0.652] | 0.447 | 0.203 | 0.468 | 0.442 | local 4090 |
| interp_gt_s1_n_s2 (CONTROL, teacher-free) | interpolated expert masks → boxes (191,613 imgs) | 2 | 2 / 17 | 0.552 [0.268, 0.730] | 0.551 | 0.256 | 0.555 | 0.541 | the cluster 3172380 |
| interp_gt_s1_n_s3 (CONTROL, teacher-free) | same | 3 | 3 / 18 | 0.499 [0.224, 0.677] | 0.471 | 0.206 | 0.500 | 0.460 | the cluster 3172381 |
| kd_bfi_x_n_s1 (T0 native KD) | canonical labels + ultralytics feature KD (teacher bfi yolo26x e21, dis 6.0) | 1 | 17 / 67 | 0.461 [0.2174, 0.6837] | 0.421 | 0.204 | 0.469 | 0.433 | local 4090 |
| kd_bfi_x_n_s2 (T0 native KD) | canonical labels + ultralytics feature KD (teacher bfi yolo26x e21, dis 6.0) | 2 | 21 / 71 | 0.489 [0.227, 0.695] | 0.441 | 0.210 | 0.527 | 0.497 | the cluster 3172397 |
| kd_bfi_x_n_s3 (T0 native KD) | same | 3 | 22 / 72 | 0.492 [0.215, 0.687] | 0.446 | 0.218 | 0.515 | 0.489 | the cluster 3172398 |
| t1s1_ft_canon_bfi_n_s1 (two-stage, NAIVE) | init = T1 s1 best.pt → expert-only fine-tune, default schedule | 1 | 1 / 51 | 0.445 [0.1774, 0.6414] | 0.396 | 0.167 | 0.463 | 0.429 | the cluster 3173205 — NEGATIVE (full-LR restart collapses the T1 solution); low-LR variant queued |
| t1_nano_k025_t015_s1_n_s1 | T1 teacher view, tau 0.15 (recall-leaning labels) | 1 | 3 / 18 | 0.526 [0.286, 0.687] | 0.508 | 0.217 | 0.531 | 0.516 | the cluster 3173149 |
| t1_nano_k025_t035_s1_n_s1 | T1 teacher view, tau 0.35 (precision-leaning labels) | 1 | 2 / 17 | 0.535 [0.295, 0.694] | 0.513 | 0.226 | 0.543 | 0.515 | the cluster 3173240 |
| t1s3_ftlow_canon_bfi_n_s3 (two-stage, LOW-LR) | init = T1 s3 best.pt → expert-only, SGD lr0 0.001, no warm-up, 40/10 | 3 | 5 / 15 | 0.522 [0.284, 0.699] | 0.503 | 0.222 | 0.510 | 0.488 | the cluster 3173800+3173980 |
| t1_nano_k025_t024_s1_keepempty_n_s1 | T1 teacher view, rip-video frames w/o confident box KEPT as negatives | 1 | 2 / 17 | 0.506 [0.253, 0.702] | 0.488 | 0.217 | 0.510 | 0.488 | the cluster 3173292 |
| t1s1_ftlow_canon_bfi_n_s1 (two-stage, LOW-LR) | init = T1 s1 best.pt → expert-only, lr0 0.001 | 1 | 6 / 16 | 0.480 [0.240, 0.672] | 0.435 | 0.207 | 0.494 | 0.454 | the cluster 3173782+3174044 |
| t1s2_ftlow_canon_bfi_n_s2 (two-stage, LOW-LR) | init = T1 s2 best.pt → expert-only, lr0 0.001 | 2 | 5 / 15 | 0.513 [0.263, 0.687] | 0.484 | 0.227 | 0.535 | 0.497 | the cluster 3173783+3174045 |
| kd_t1_n_s1 (dense + native KD) | T1 headline view + ultralytics feature KD (bfi yolo26x, dis 6.0) | 1 | 3 / 18 | 0.495 [0.249, 0.643] | 0.433 | 0.164 | 0.505 | 0.459 | the cluster 3173350 (12.8 h) |
| kd_t1_n_s2 (dense + native KD) | same | 2 | 3 / 18 | 0.505 [0.229, 0.686] | 0.493 | 0.224 | 0.524 | 0.517 | the cluster 3173939 |
| kd_t1_n_s3 (dense + native KD) | same | 3 | 2 / 17 | 0.535 [0.297, 0.691] | 0.522 | 0.205 | 0.540 | 0.527 | the cluster 3174088 |
| kd_interp_n_s2 (interp control + native KD) | interp view + ultralytics feature KD (bfi yolo26x, dis 6.0) | 2 | 2 / 17 | 0.500 [0.226, 0.681] | 0.488 | 0.221 | 0.499 | 0.481 | the cluster 3173962 |
| kd_interp_n_s3 (interp control + native KD) | same | 3 | 3 / 18 | 0.502 [0.248, 0.673] | 0.482 | 0.208 | 0.521 | 0.498 | the cluster 3174089 |
| t1_nano_k025_t024_s1_noneg_n_s1 | T1 teacher view WITHOUT the no-rip videos' dense negatives (136,813 imgs) | 1 | 4 / 19 | 0.499 [0.238, 0.682] | 0.473 | 0.206 | 0.521 | 0.493 | the cluster 3174101 |
| t1_nano_k025_t024_s1_noneg_n_s2 | same | 2 | 3 / 18 | 0.467 [0.201, 0.664] | 0.421 | 0.191 | 0.485 | 0.453 | the cluster 3177883 |
| t1_nano_k025_t024_s1_noneg_n_s3 | same | 3 | 3 / 18 | 0.538 [0.292, 0.696] | 0.510 | 0.242 | 0.548 | 0.526 | the cluster 3178449 |
| t1_nanoadd_k025_t024_s1_n_s1 | teacher = RF-DETR-Seg-nano **+additional** (disclosed scenario), same recipe | 1 | 24 / 39 | 0.540 [0.283, 0.711] | 0.511 | 0.236 | 0.541 | 0.506 | the cluster 3173293+3173960+3173972 (RESUME chain ×2 after the OOM/quota kills; late best epoch) |
| t1_nano_k0_t024_s1_n_s1 | T1 teacher view, NO temporal smoothing (k=0) | 1 | 2 / 17 | 0.497 [0.247, 0.687] | 0.484 | 0.216 | 0.512 | 0.501 | the cluster 3172401 |
| t1_nano_k0_t024_s1_n_s2 | same | 2 | 2 / 17 | 0.491 [0.248, 0.651] | 0.449 | 0.204 | 0.495 | 0.458 | the cluster 3177884 |
| t1_nano_k0_t024_s1_n_s3 | same | 3 | 2 / 17 | 0.547 [0.287, 0.707] | 0.537 | 0.245 | 0.537 | 0.523 | the cluster 3178636 |
| t1_nano_k025_t024_s2_n_s1 | T1 teacher view, stride 2 (108,387 imgs) | 1 | 3 / 18 | 0.534 [0.265, 0.707] | 0.515 | 0.187 | 0.537 | 0.519 | the cluster 3172402 (local recompute identical) |
| t1_nano_k025_t024_s2_n_s3 | T1 teacher view, stride 2 | 3 | 2 / 17 | 0.541 [0.2932, 0.7084] | 0.520 | 0.194 | 0.509 | 0.493 | the cluster 3173402 |
| t1_nano_k025_t024_s2_n_s2 | T1 teacher view, stride 2 | 2 | 25 / 40 | 0.484 [0.225, 0.680] | 0.444 | 0.188 | 0.496 | 0.449 | the cluster 3173387+3173974 (RESUME chain after the quota kill) |
| canon_bfi_s_s1 (yolo26s, edge ladder) | canonical expert labels, **yolo26s** student | 1 | 1 / 51 | 0.555 [0.349, 0.709] | 0.528 | 0.181 | 0.547 | 0.547 | the cluster 3174090 (degenerate best-epoch-1 selection, as in the benchmark's s/m rows) |
| canon_bfi_s_s2 (yolo26s, edge ladder) | canonical expert labels, yolo26s student | 2 | 25 / 75 | 0.453 [0.209, 0.664] | 0.375 | 0.167 | 0.482 | 0.444 | the cluster 3177688 |
| canon_bfi_s_s3 (yolo26s, edge ladder) | same | 3 | 17 / 67 | 0.430 [0.141, 0.655] | 0.401 | 0.150 | 0.466 | 0.450 | local 4090 |
| t1_nano_k025_t024_s1_s_s1 (yolo26s on the DENSE view) | T1 teacher view, **yolo26s** student | 1 | 1 / 16 | 0.567 [0.371, 0.716] | 0.519 | 0.230 | 0.583 | 0.554 | the cluster 3174091 (8.3 h) |
| t1_nano_k025_t024_s1_s_s2 (yolo26s on the DENSE view) | same | 2 | 1 / 16 | 0.537 [0.330, 0.680] | 0.477 | 0.205 | 0.548 | 0.507 | the cluster 3177689 |
| t1_nano_k025_t024_s1_s_s3 (yolo26s on the DENSE view) | same | 3 | 3 / 18 | 0.545 [0.315, 0.725] | 0.520 | 0.242 | 0.540 | 0.507 | local 4090 |
| t1_nano_k025_t024_s4_n_s1 | T1 teacher view, stride 4 (69,609 imgs) | 1 | 4 / 19 | 0.465 [0.189, 0.667] | 0.463 | 0.210 | 0.467 | 0.467 | the cluster 3173090 |
| t1_nano_k025_t024_s4_n_s2 | T1 teacher view, stride 4 | 2 | 10 / 25 | 0.516 [0.2764, 0.6881] | 0.482 | 0.246 | 0.521 | 0.501 | the cluster 3173418 |
| t1_nano_k025_t024_s4_n_s3 | T1 teacher view, stride 4 | 3 | 24 / 39 | 0.544 [0.277, 0.725] | 0.530 | 0.235 | 0.545 | 0.529 | the cluster 3173582+3173976 (RESUME chain) |


### Aggregated by variant (mean ± std over seeds; regenerated 2026-08-27 09:50 from `summarize_students.py` -> `results/student_summary_by_variant.tsv`)
| variant | n seeds | best epochs | bfi F2@50 | bfi AP50 | bfi AP50-95 | tight F2@50 | tight AP50 |
|---|---|---|---|---|---|---|---|
| canonical expert labels (T0 baseline), yolo26n | 3 (1,2,3) | 18,8,22 | 0.483 ± 0.022 | 0.444 ± 0.031 | 0.191 ± 0.019 | 0.500 ± 0.026 | 0.465 ± 0.036 |
| canonical expert labels, **yolo26s** | 3 (1,2,3) | 1,25,17 | 0.479 ± 0.054 | 0.435 ± 0.067 | 0.166 ± 0.012 | 0.498 ± 0.035 | 0.480 ± 0.047 |
| interpolated-expert control (teacher-free), yolo26n | 3 (1,2,3) | 2,2,3 | 0.507 ± 0.033 | 0.490 ± 0.045 | 0.222 ± 0.024 | 0.508 ± 0.036 | 0.481 ± 0.043 |
| canonical + native feature KD (T0), yolo26n | 3 (1,2,3) | 17,21,22 | 0.481 ± 0.014 | 0.436 ± 0.011 | 0.211 ± 0.006 | 0.504 ± 0.025 | 0.473 ± 0.028 |
| interp control + native feature KD | 2 (2,3) | 2,3 | 0.501 ± 0.001 | 0.485 ± 0.003 | 0.214 ± 0.007 | 0.510 ± 0.011 | 0.489 ± 0.009 |
| dense (T1 view) + native feature KD | 3 (1,2,3) | 3,3,2 | 0.512 ± 0.017 | 0.482 ± 0.037 | 0.198 ± 0.025 | 0.523 ± 0.014 | 0.501 ± 0.030 |
| T1 view, tau 0.15 | 1 (1) | 3 | 0.526 | 0.508 | 0.217 | 0.531 | 0.516 |
| T1 view, keep empty rip-video frames | 1 (1) | 2 | 0.506 | 0.488 | 0.217 | 0.510 | 0.488 |
| **dense supervision, T1 teacher view (headline), yolo26n** | 3 (1,2,3) | 2,2,2 | 0.531 ± 0.016 | 0.505 ± 0.026 | 0.232 ± 0.013 | 0.532 ± 0.008 | 0.504 ± 0.020 |
| T1 view, no dense negatives | 3 (1,2,3) | 4,3,3 | 0.501 ± 0.029 | 0.468 ± 0.037 | 0.213 ± 0.022 | 0.518 ± 0.026 | 0.491 ± 0.030 |
| **dense supervision (T1 view), yolo26s** | 3 (1,2,3) | 1,1,3 | 0.550 ± 0.013 | 0.505 ± 0.020 | 0.226 ± 0.015 | 0.557 ± 0.018 | 0.523 ± 0.022 |
| T1 view, stride 2 | 3 (1,2,3) | 3,25,2 | 0.520 ± 0.025 | 0.493 ± 0.035 | 0.190 ± 0.003 | 0.514 ± 0.017 | 0.487 ± 0.029 |
| T1 view, stride 4 | 3 (1,2,3) | 4,10,24 | 0.509 ± 0.033 | 0.492 ± 0.028 | 0.230 ± 0.015 | 0.511 ± 0.033 | 0.499 ± 0.025 |
| T1 view, tau 0.35 | 1 (1) | 2 | 0.535 | 0.513 | 0.226 | 0.543 | 0.515 |
| T1 view, no temporal smoothing | 3 (1,2,3) | 2,2,2 | 0.512 ± 0.025 | 0.490 ± 0.036 | 0.222 ± 0.017 | 0.515 ± 0.017 | 0.494 ± 0.027 |
| T1 view from the +additional-data teacher | 1 (1) | 24 | 0.540 | 0.511 | 0.236 | 0.541 | 0.506 |
| two-stage naive (dense init → expert, lr0 0.01) | 1 (1) | 1 | 0.445 | 0.396 | 0.167 | 0.463 | 0.429 |
| two-stage low-LR (dense init → expert, lr0 0.001) | 3 (1,2,3) | 6,5,5 | 0.505 ± 0.018 | 0.474 ± 0.028 | 0.219 ± 0.009 | 0.513 ± 0.017 | 0.479 ± 0.019 |

Density curve (train images / bfi F2, 3 seeds each): canonical 31k / 0.483±0.022 → stride 4 70k / 0.509±0.033 → stride 2 108k / 0.520±0.025 → full 191k / 0.531±0.016 — monotone in density. CAVEAT: the two RESUME-chained runs (stride-2 s2, stride-4 s3; restarted after the quota kill) show late best epochs (25, 24) unlike every uninterrupted dense run (2–4); their scores (0.484, 0.544) straddle their siblings — treated as valid but flagged.
(updated 2026-08-26 11:30; canonical, T1 and interp control complete at 3 seeds; KD s1, ablation seeds, low-LR two-stage, KD+dense pending; canonical s1 + interp s1 local, native KD, ablations, two-stage pending)

**Finding (first stated 2026-08-26 at 2 seeds; the 3-seed reading is in the paired-delta section below):** the teacher-free control — expert masks
SDF-interpolated between annotated frames, boxed, same no-rip negatives — matches the teacher view
(0.525 ± 0.027 vs 0.532 ± 0.020 F2; AP50 0.511 vs 0.508; AP50-95 equal). Supervision DENSITY explains the
gain over canonical (+0.05 F2); the teacher adds little on top on the annotated train videos. The teacher's
residual value must come from (a) frames outside annotated pairs / unlabeled footage, (b) feature-level KD,
or (c) combination/schedules — to be tested; report as-is otherwise.

**Why they match (label agreement, 2026-08-26 04:00):** on the 105,724 rip-video frames both views label, the
teacher's smoothed pseudo-boxes reproduce the interpolated expert boxes at P 0.985 / R 0.988 (IoU ≥ 0.5), matched
median IoU 0.915 (p10 0.78, p90 0.97); frame sets differ by 6 vs 361 frames. The two views are near-identical label
sets, so equal student results are expected. Interpretation: on its own training videos the teacher essentially
re-emits the (interpolated) expert supervision it was trained on; a teacher only adds information on footage the
experts never labelled — which this benchmark cannot provide without leaking val/test videos.

### Seed-pooled paired deltas (per-video TP/FP/FN summed over each variant's seeds, one video-level bootstrap, B=1000)
| comparison | seeds | bfi dF2@50 [95% CI] | tight dF2@50 [95% CI] |
|---|---|---|---|
| **dense supervision (T1 teacher) − canonical** | 3 vs 3 | **+0.047 [−0.013, +0.104]** | +0.032 [−0.022, +0.079] |
| interp control (teacher-free) − canonical | 3 vs 3 | +0.023 [−0.068, +0.101] | +0.007 [−0.065, +0.061] |
| native feature KD − canonical | 3 vs 3 | -0.0030 [-0.0400, +0.0342] | +0.0039 [-0.0358, +0.0517] |
| T1 teacher − interp control | 3 vs 3 | +0.024 [−0.019, +0.105] | +0.025 [−0.014, +0.095] |
| two-stage low-LR − T1 teacher (its init) | 3 vs 3 | −0.026 [−0.074, +0.026] | −0.019 [−0.066, +0.027] |
| dense + native KD − dense (T1 teacher) | 3 vs 3 | −0.019 [−0.089, +0.029] | −0.009 [−0.071, +0.036] |
| **dense (T1 teacher) − dense WITHOUT the no-rip videos' negatives** | 3 vs 3 | **+0.029 [+0.001, +0.069]** | +0.014 [−0.017, +0.050] |
| **yolo26s dense − yolo26s canonical** | 3 vs 3 | **+0.069 [+0.008, +0.136]** | **+0.058 [+0.006, +0.112]** |
| dense (T1 teacher) − dense WITHOUT temporal smoothing | 3 vs 3 | +0.020 [−0.009, +0.043] | +0.018 [−0.003, +0.036] |
Reading (3 seeds each, 2026-08-26 11:30): teacher-labelled dense supervision is a consistent +0.05 F2 over canonical
(every same-seed pair positive; pooled CI touches zero at n = 71 videos). The teacher-free interpolation control is
noisier (seeds 0.552 / 0.499 / 0.471) and lands +0.02 over canonical and −0.02 under the teacher view — both inside
the CI. Honest statement: densification explains most of the gain; the teacher view is at least as good and less
seed-sensitive (std 0.016 vs 0.033), plausibly because its labels also cover frames outside annotated pairs and are
temporally smoothed; label sets agree at P/R 0.985/0.988, so a large teacher-specific gain is impossible here.

### Paired deltas (video-level bootstrap, B=1000, seed 0; `results/student_bootstrap_cis.tsv`)
| comparison | GT | dF2@50 [95% CI] | note |
|---|---|---|---|
| T1 stride-2 s1 (e3) − canonical s2 (e8) | bfi | +0.033 [−0.031, +0.078] | single seed each; CI includes 0 |
| same | tight | +0.019 [−0.032, +0.060] | |
| T1 full s1 (e2) − canonical s2 (e8) | bfi | +0.012 [−0.056, +0.068] | single seed each |
| same | tight | +0.009 [−0.053, +0.069] | |
| T1 stride-2 s1 − T1 full s1 | bfi | +0.021 [−0.032, +0.078] | half density ≥ full density so far |
| **T1 full s3 (e2) − canonical s3 (e22)** | bfi | **+0.100 [+0.003, +0.212]** | same seed; first CI excluding 0 |
| same | tight | +0.081 [−0.007, +0.173] | |
| interp control s2 (e2) − canonical s2 (e8) | bfi | +0.051 [−0.007, +0.092] | same seed |
| interp control s3 (e3) − canonical s3 (e22) | bfi | +0.046 [−0.058, +0.165] | same seed |
| T1 teacher s3 (e2) − interp control s3 (e3) | bfi | +0.053 [+0.000, +0.137] | same seed; but interp s2 (0.552) > teacher s1 (0.513) — seed-dominated |
| same | tight | +0.044 [−0.008, +0.129] | |
| T1 teacher s2 (e2) − canonical s2 (e8) | bfi | +0.027 [−0.040, +0.074] | same seed |
| T1 teacher s2 (e2) − interp control s2 (e2) | bfi | −0.025 [−0.080, +0.041] | same seed; teacher vs interp: +0.053 (s3), −0.025 (s2) → no consistent difference |
| T1 smoothed s1 − T1 unsmoothed (k=0) s1 | bfi | +0.0164 [-0.0311, +0.0653] | same seed |
| same | tight | +0.0150 [-0.0281, +0.0592] | |
| T1 smoothed s2 − T1 unsmoothed (k=0) s2 | bfi | +0.036 [−0.010, +0.078] | same seed; 2 seeds: unsmoothed 0.494 ± 0.003 vs smoothed 0.531 ± 0.016 → temporal smoothing ≈ +0.03 F2 (final 3-seed reading: ≈ +0.02, consistent direction, CI crosses 0) |
| T1 smoothed s3 − T1 unsmoothed (k=0) s3 | bfi | +0.005 [−0.028, +0.028] | seed 3; 3 seeds: unsmoothed 0.512 ± 0.025 vs 0.531 ± 0.016, pooled +0.020 [−0.009, +0.043] — smoothing helps on every seed (+0.016 / +0.036 / +0.005) but the pooled CI crosses 0 |
| native KD s2 (e21) − canonical s2 (e8) | bfi | −0.012 [−0.126, +0.092] | same seed; KD helps AP50-95 (+0.01), not F2 |
| native KD s2 − T1 teacher s2 | bfi | −0.039 [−0.151, +0.079] | |
| native KD s3 (e22) − canonical s3 (e22) | bfi | +0.039 [−0.043, +0.141] | same seed; KD over canonical: −0.012 (s2), +0.039 (s3) → mean +0.013, AP50-95 +0.03 consistent |
| native KD s1 (e17) − canonical s1 (e18) | bfi | -0.0359 [-0.0835, +0.0055] | same seed; KD over canonical per seed: -0.0359 / −0.012 / +0.039 → null on F2, +0.02 AP50-95 |
| T1 teacher s1 (e2) − canonical s1 (e18) | bfi | +0.016 [−0.046, +0.078] | same seed; T1 − canonical over the 3 same-seed pairs: +0.016 / +0.027 / +0.100 (mean +0.048) |
| interp control s1 (e2) − canonical s1 (e18) | bfi | −0.026 [−0.150, +0.090] | same seed (the weak interp seed) |
| T1 teacher s1 (e2) − interp control s1 (e2) | bfi | +0.042 [−0.034, +0.151] | same seed; tight: +0.059 [+0.018, +0.151] |
| low-LR two-stage s3 (e5) − its T1 init s3 (e2) | bfi | −0.030 [−0.083, +0.034] | expert-only fine-tune does not add to the dense solution (tight −0.034) |
| low-LR two-stage s3 − canonical s3 | bfi | +0.069 [−0.029, +0.182] | keeps most of the dense gain |
| low-LR two-stage s2 (e5) − its T1 init s2 (e2) | bfi | −0.014 [−0.075, +0.046] | schedule lever: null on both seeds |
| keep-empty s1 − headline s1 (drop-empty) | bfi | −0.007 [−0.091, +0.050] | dropping teacher-miss frames is not what drives the gain |
| low-LR two-stage s1 (e6) − its T1 init s1 (e2) | bfi | −0.033 [−0.085, +0.013] | schedule lever CLOSED: −0.033 / −0.014 / −0.030 on the three seeds |
| dense + KD s1 (e3) − dense s1 (e2) | bfi | −0.019 [−0.118, +0.061] | KD on the dense view does not stack (AP50-95 0.164 vs 0.216); seeds 2, 3 running |
| dense + KD s2 (e3) − dense s2 (e2) | bfi | −0.022 [−0.105, +0.025] | second seed agrees: no stacking (tight −0.002) |
| dense + KD s3 (e2) − dense s3 (e2) | bfi | −0.017 [−0.064, +0.033] | third seed: −0.019 / −0.022 / −0.017 → KD does not stack (3-seed 0.512 ± 0.017 vs 0.531 ± 0.016) |
| interp + KD s2 (e2) − interp s2 (e2) | bfi | **−0.051 [−0.092, −0.022]** | KD significantly HARMS the densified student (tight −0.057 [−0.100, −0.029]) |
| interp + KD s3 (e3) − interp s3 (e3) | bfi | +0.003 [−0.031, +0.065] | vs the weak interp seed (0.499): KD pins the student at ≈0.50 whatever the densified baseline (2-seed 0.501 ± 0.001) |
| no-negatives s1 (e4) − headline s1 (e2) | bfi | −0.014 [−0.045, +0.009] | seed 1 |
| no-negatives s2 (e3) − headline s2 (e2) | bfi | −0.060 [−0.148, +0.007] | seed 2; tight −0.041 [−0.092, −0.004]. Two seeds: without the no-rip videos' dense negatives 0.483 ± 0.016 vs 0.531 ± 0.016 → the video-level negatives ARE part of the gain (≈ −0.05 without them) |
| no-negatives s3 (e3) − headline s3 (e2) | bfi | −0.014 [−0.065, +0.042] | seed 3; 3 seeds without negatives 0.501 ± 0.029 vs 0.531 ± 0.016; pooled +0.029 [+0.001, +0.069] for keeping them — the video-level negatives are a significant recipe component |
| yolo26s dense s1 (e1) − yolo26s canonical s1 (e1) | bfi | +0.012 [−0.050, +0.076] | dense gain transfers to the s-model mainly at strict IoU (AP50-95 0.230 vs 0.181) and on tight GT (+0.036); seeds 2–3 pending |
| yolo26s dense s2 (e1) − yolo26s canonical s2 (e25) | bfi | +0.084 [−0.013, +0.187] | 2 seeds: dense yolo26s 0.552 ± 0.015 vs canonical 0.504 ± 0.051 — dense supervision helps the s-model too and removes its seed instability |
| yolo26s dense s3 (e3) − yolo26s canonical s3 (e17) | bfi | +0.115 [+0.010, +0.269] | 3 seeds: dense 0.550 ± 0.013 vs canonical 0.479 ± 0.054; pooled +0.069 [+0.008, +0.136] — the dense gain on the s-model is significant and it stabilizes training |
| +additional-teacher view s1 (e24) − canonical-teacher view s1 (e2) | bfi | +0.027 [−0.049, +0.113] | stronger teacher (0.722 vs 0.697 box F2) → student +0.03, within noise; tight +0.014 |

## Per-stratum test results (2026-08-27; `stratified_results.py` → `results/stratified_test.tsv`)

Per-video TP/FP/FN on test (bfi GT, conf frozen on val per seed, evaluator's matcher), seeds micro-pooled per
variant, joined with the RipBench per-video metadata sheet; paired delta with a video-level bootstrap
inside each stratum (B=1000, seed 0). All 71 test videos have a Viewpoint; camera motion / shoreline fields are
blank for most no-rip videos ("n/a").

| stratum | videos (rip) | GT boxes | yolo26n canon | interp | dense | yolo26s canon | dense | dense − canon (n) | dense − interp (n) |
|---|---|---|---|---|---|---|---|---|---|
| viewpoint aerial bird's eye | 17 (7) | 7,965 | 0.670 | 0.699 | 0.711 | 0.658 | 0.680 | +0.041 [+0.000, +0.098] | +0.012 [+0.000, +0.024] |
| viewpoint aerial tilted | 27 (13) | 5,703 | 0.260 | 0.181 | 0.273 | 0.256 | 0.353 | +0.013 [−0.076, +0.098] | +0.092 [−0.049, +0.317] |
| viewpoint elevated beachfront | 22 (8) | 3,966 | 0.465 | 0.624 | 0.591 | 0.503 | 0.623 | +0.126 [−0.069, +0.198] | −0.033 [−0.055, +0.042] |
| viewpoint water-level beachfront | 5 (3) | 897 | 0.219 | 0.166 | 0.157 | 0.200 | 0.384 | −0.062 [−0.090, +0.026] | −0.010 [−0.023, −0.001] |
| camera static | 9 (9) | 7,434 | 0.728 | 0.681 | 0.729 | 0.719 | 0.702 | +0.002 [−0.082, +0.073] | +0.048 [−0.064, +0.256] |
| camera translation (drone) | 17 (17) | 9,618 | 0.353 | 0.436 | 0.442 | 0.364 | 0.499 | +0.090 [−0.011, +0.158] | +0.006 [−0.027, +0.043] |
| camera shake | 4 (4) | 1,029 | 0.159 | 0.101 | 0.107 | 0.162 | 0.357 | −0.053 [−0.083, +0.022] | +0.006 [−0.007, +0.048] |
| signature dark gap | 20 (20) | 10,824 | 0.584 | 0.614 | 0.650 | 0.591 | 0.635 | +0.066 [−0.010, +0.151] | +0.036 [−0.026, +0.215] |
| signature sediment plume | 6 (6) | 6,078 | 0.362 | 0.392 | 0.414 | 0.361 | 0.456 | +0.052 [−0.011, +0.096] | +0.023 [+0.000, +0.029] |
| signature breaking-wave gap | 4 (4) | 1,020 | 0.287 | 0.230 | 0.230 | 0.267 | 0.426 | −0.057 [−0.085, −0.007] | +0.000 [−0.022, +0.048] |
| ALL | 71 (31) | 18,531 | 0.484 | 0.507 | 0.531 | 0.481 | 0.550 | +0.047 [−0.014, +0.101] | +0.024 [−0.019, +0.094] |

Reading: the density gain concentrates on the deployment-relevant strata that were NOT already solved — elevated
beach cameras (+0.126, n=22) and moving drone footage (+0.090, n=17) — and on dark-gap / sediment-plume rips;
static cameras were already at F2 0.73 and gain nothing; the small hard strata (water-level, shake, breaking-wave
gap; 4–5 videos each) get slightly worse for yolo26n while yolo26s dense improves them (0.36–0.43 vs 0.16–0.27).
Strata are small and every CI but bird's-eye/breaking-wave crosses 0: report as a breakdown, not as separate claims.

## Event-level deployment metrics (2026-08-27; `event_metrics.py` → `results/event_metrics.tsv`, 384 rows)

Test videos replayed as streams from the dumps (sampled frames = every 6th native frame; per-video fps from ffprobe,
`manifests/{val,test}_video_fps.tsv`), detector at its VAL-frozen frame-level conf, optional cls gate (yolo26n-cls 224,
thr 0.50 = `cascade_sim.tsv`), duty = frames kept per second, k-of-n persistence. Alarm on a rip video = any detection
(TP or FP box) at/after its first expert-annotated rip frame; a no-rip video "alarms" if it has ≥ 1 false episode
(positives merged within 10 s). Only **0.28 h** of no-rip test footage exists (40 clips, ~25 s each), so episodes/hour
are indicative; the fraction of no-rip clips with a false episode is the robust number. Video-F2 = F2 of the per-video
alarm decision (rip alarmed = TP, no-rip alarmed = FP, rip silent = FN). Mean ± std over 3 seeds, 1-of-1 persistence.

| student | gate | duty | rip videos alarmed | latency p90 (s) | no-rip clips alarmed | video-F2 | detector runs (frac of frames) |
|---|---|---|---|---|---|---|---|
| yolo26n canonical | off | 1 fps | 0.76 ± 0.04 | 3.7 ± 1.2 | 0.36 ± 0.05 | 0.73 ± 0.04 | 0.21 |
| yolo26n canonical | off | 0.2 fps | 0.67 ± 0.02 | 7.6 ± 3.6 | 0.18 ± 0.04 | 0.68 ± 0.02 | 0.04 |
| yolo26n canonical | 0.50 | 1 fps | 0.72 ± 0.04 | 4.1 ± 0.8 | 0.15 ± 0.00 | 0.73 ± 0.04 | 0.10 |
| yolo26n canonical | 0.50 | 0.2 fps | 0.63 ± 0.03 | 5.5 ± 0.7 | 0.10 ± 0.00 | 0.67 ± 0.03 | 0.02 |
| yolo26n dense | off | 1 fps | 0.70 ± 0.04 | 5.5 ± 1.1 | 0.35 ± 0.02 | 0.68 ± 0.04 | 0.21 |
| yolo26n dense | off | 0.2 fps | 0.65 ± 0.07 | 8.9 ± 0.9 | 0.21 ± 0.02 | 0.66 ± 0.07 | 0.04 |
| yolo26n dense | 0.50 | 1 fps | 0.66 ± 0.03 | 5.1 ± 1.0 | 0.10 ± 0.00 | 0.69 ± 0.03 | 0.10 |
| yolo26n dense | 0.50 | 0.2 fps | 0.62 ± 0.06 | 10.2 ± 4.9 | 0.07 ± 0.01 | 0.66 ± 0.05 | 0.02 |
| yolo26s dense | off | 1 fps | 0.76 ± 0.03 | 2.3 ± 0.8 | 0.52 ± 0.03 | 0.70 ± 0.02 | 0.21 |
| yolo26s dense | 0.50 | 1 fps | 0.68 ± 0.03 | 2.3 ± 1.8 | 0.17 ± 0.03 | 0.69 ± 0.03 | 0.10 |
| yolo26s dense | 0.50 | 0.2 fps | 0.66 ± 0.02 | 2.0 ± 2.1 | 0.13 ± 0.00 | 0.68 ± 0.01 | 0.02 |

Median time-to-alarm is 0 s in every setting (the alarm fires on the first annotated rip frame); p90 is 2–10 s.
An event-level operating point selected on val (conf maximising video-F2 at gate off / 1 fps) lands within ±0.05 of
the frame-level conf, is seed-unstable (0.01–0.13) and changes nothing materially (rows `operating_point=event-F2`);
the paper keeps the single frame-level operating point.

**Finding (honest, 3 seeds): the dense-supervision gain is a FRAME-level gain, not a VIDEO-level one.** The dense
yolo26n alarms on 0.70 ± 0.04 of the rip videos vs 0.76 ± 0.04 for the canonical model (1 fps, gate off; video-F2 0.68
vs 0.73) with the same false-alarm rate; dense supervision raises per-frame recall inside the videos the model already
detects (the per-video skew in the Limitations) rather than unlocking new videos. What makes deployment viable is the
gate: it cuts the no-rip clips with a false episode from 0.35–0.52 to 0.10–0.17 at a cost of ~4–8 points of rip-video
recall, with the detector running on 10 % (1 fps) or 2 % (0.2 fps) of the frames. Duty cycling from 1 to 0.2 fps costs
4–9 points of rip-video recall and 2–5 s of p90 latency. To be stated in the paper as such.

## Inference-time temporal smoothing of student outputs (2026-08-27; `smooth_student_outputs.py`, rows `*_sm1` / `*_sm2` in `results/student_eval_{bfi,tight}.tsv`)

The students run frame by frame; the deployment stream gives free temporal context. The SAME IoU-linked window
smoothing used for the teacher labels (score = mean over the ±k window with unmatched frames counting 0, box =
score-weighted mean, link IoU 0.5) is applied to the student's post-NMS dumps along each video's sampled-frame
sequence (every 6th native frame: k=1 ≈ ±0.2 s, k=2 ≈ ±0.4 s), followed by NMS 0.7 again (without it the
sub-threshold duplicates of a strong box inherit its score and FPs double — the first pass showed 0.50 → 0.46 for
exactly that reason; k=0 reproduces the original dumps bit-exactly). Re-scored with the benchmark evaluator
(val-swept conf → frozen → test). Mean ± std over 3 seeds; per-seed deltas in brackets.

| student | GT view | F2 raw | F2 ±1 frame | F2 ±2 frames | ΔF2 ±2 per seed | AP50 raw → ±2 |
|---|---|---|---|---|---|---|
| yolo26n canonical | bfi | 0.483 ± 0.022 | 0.507 ± 0.025 | 0.508 ± 0.026 | +0.030 / +0.023 / +0.019 | 0.444 → 0.464 |
| yolo26n interp | bfi | 0.507 ± 0.033 | 0.534 ± 0.035 | 0.539 ± 0.037 | +0.033 / +0.039 / +0.025 | 0.490 → 0.512 |
| yolo26n dense | bfi | 0.531 ± 0.016 | 0.557 ± 0.017 | **0.560 ± 0.017** | +0.024 / +0.042 / +0.021 | 0.505 → 0.527 |
| yolo26s dense | bfi | 0.550 ± 0.013 | 0.570 ± 0.016 | **0.575 ± 0.017** | +0.030 / +0.019 / +0.027 | 0.505 → 0.526 |
| yolo26n canonical | tight | 0.500 ± 0.026 | 0.527 ± 0.033 | 0.531 ± 0.037 | +0.043 / +0.032 / +0.016 | 0.465 → 0.484 |
| yolo26n dense | tight | 0.532 ± 0.008 | 0.561 ± 0.009 | 0.562 ± 0.008 | +0.025 / +0.047 / +0.019 | 0.504 → 0.529 |
| yolo26s dense | tight | 0.557 ± 0.018 | 0.575 ± 0.020 | 0.580 ± 0.022 | +0.028 / +0.020 / +0.021 | 0.523 → 0.545 |

Reading: +0.02–0.03 F2 and +0.02 AP50 for every student, every seed and both GT views (24/24 paired deltas
positive), on top of dense supervision (0.531 → 0.560 for yolo26n) — at zero model cost; the price is a k-frame
output delay (0.2–0.4 s at the 5 fps sampled rate; k s at a 1 fps duty cycle, where the neighbour spacing is 1 s
and the effect is untested). ±2 ≈ ±1. Video-level bootstrap CIs for k=1 (the paper's a-priori window; seed-pooled, B=1000, bfi GT): canonical +0.023 [+0.017, +0.033], interp +0.027 [+0.019, +0.034], dense yolo26n +0.026 [+0.013, +0.042], dense yolo26s +0.021 [+0.011, +0.030] — all exclude 0. For k=2: canonical +0.025 [+0.015, +0.039], interp +0.033 [+0.023, +0.042], dense yolo26n +0.029 [+0.015, +0.045], dense yolo26s +0.025 [+0.016, +0.036] — every interval excludes 0; smoothed dense − smoothed canonical +0.051 [-0.022, +0.115] (the density effect is unchanged by smoothing).

## Semi-supervised holdout — does the teacher matter when labels are missing? (started 2026-08-27; design `ICRA_2027/notes/design_semisup_holdout.md`)

Video-level 50 % holdout of the expert labels inside the s1.3 TRAIN split (`manifests/holdout_h50_seed0.tsv`: 83 labelled /
82 unlabelled videos, stratified rip / no-rip, seed 0); val/test untouched; conditions (a) sparse = expert labels of the
labelled half (19,032 images), (b) interp = interpolated expert labels of the labelled half (115,624), (c) teacher-dense =
T1 recipe on ALL 165 videos with a teacher retrained on the labelled half only (expert frames enter only from labelled
videos). yolo26n, 3 seeds each, benchmark evaluator, bfi GT (tight in the tight TSV). (c-drop) = (c) with the teacher-empty frames of the unlabelled videos DROPPED (152,684 images; isolates the ~17 % wrong negatives the
label-free recipe injects on rip footage; 1 seed). Rows land here as they complete
(provenance: `results/student_eval_{bfi,tight}.tsv`, `results/student_train_summary.tsv`, cluster job ids in `ICRA_2027/JOBS.md`).

| condition | seed | best / stopped epoch | test F2 (bfi) | AP50 | AP50-95 | F2 (tight) | job |
|---|---|---|---|---|---|---|---|
| (a) sparse labelled half | 1 | 35 / 85 | 0.413 | 0.340 | 0.141 | 0.425 | 3180388 eh-sp1 |
| (a) sparse labelled half | 2 | 24 / 74 | 0.419 | 0.357 | 0.137 | 0.439 | 3180410 eh-sp2 |
| (a) sparse labelled half | 3 | 31 / 81 | 0.339 | 0.304 | 0.106 | 0.376 | 3180748 eh-sp3 |
| **(a) mean ± std (3 seeds)** | | | **0.390 ± 0.036** | 0.334 | 0.128 | 0.413 | |
| (b) interp labelled half | 1 | 6 / 21 | 0.388 | 0.365 | 0.141 | 0.412 | 3180389 eh-ip1 |
| (b) interp labelled half | 2 | 4 / 19 | 0.394 | 0.343 | 0.133 | 0.418 | 3180583 eh-ip2 |
| (b) interp labelled half | 3 | 6 / 21 | 0.410 | 0.309 | 0.113 | 0.397 | 3180816 eh-ip3 |
| **(b) mean ± std (3 seeds)** | | | **0.397 ± 0.009** | 0.339 | 0.129 | 0.409 | |
| (a) sparse labelled half, yolo26s | 1 | 9 / 59 | 0.375 | 0.293 | 0.088 | 0.386 | 3181470 eh-sps1 |
| (a) sparse labelled half, yolo26s | 2 | 15 / 65 | 0.345 | 0.234 | 0.074 | 0.357 | 3182780 eh-sps2 |
| (b) interp labelled half, yolo26s | 1 | 2 / 17 | 0.431 | 0.400 | 0.171 | 0.408 | 3181471 eh-ips1 |
| (b) interp labelled half, yolo26s | 2 | 3 / 18 | 0.401 | 0.360 | 0.103 | 0.431 | 3182781 eh-ips2 |
| **yolo26s (a) / (b) mean ± std (2 seeds)** | | | **0.360 ± 0.015 / 0.416 ± 0.015** | 0.264 / 0.380 | 0.081 / 0.137 | 0.372 / 0.420 | |
| (b+) interp labelled half + video-level negatives of the unlabelled no-rip videos | 1 | 3 / 18 | 0.382 | 0.353 | 0.128 | 0.404 | 3181608 eh-bp1 |
| (b+) interp labelled half + video-level negatives of the unlabelled no-rip videos | 2 | 3 / 18 | 0.420 | 0.378 | 0.124 | 0.438 | 3181609 eh-bp2 |
| (b+) interp labelled half + video-level negatives of the unlabelled no-rip videos | 3 | 3 / 18 | 0.355 | 0.330 | 0.127 | 0.402 | 3181610 eh-bp3 |
| **(b+) mean ± std (3 seeds)** | | | **0.386 ± 0.027** | 0.354 | 0.126 | 0.415 | |
| (c) teacher-only unlabelled half + T1 labelled half | 1 | 2 / 17 | 0.472 | 0.415 | 0.190 | 0.480 | 3184509 eh-tc1 |
| (c) teacher-only unlabelled half + T1 labelled half | 2 | 2 / 17 | 0.452 | 0.373 | 0.151 | 0.472 | 3184510 eh-tc2 |
| (c) teacher-only unlabelled half + T1 labelled half | 3 | 3 / 18 | 0.451 | 0.405 | 0.180 | 0.475 | 3184511 eh-tc3 |
| **(c) mean ± std (3 seeds)** | | | **0.458 ± 0.010** | 0.398 | 0.174 | 0.476 | |
| (c-drop) = (c) with teacher-empty frames of the unlabelled videos dropped | 1 | 3 / 18 | 0.466 | 0.425 | 0.220 | 0.497 | 3184532 eh-tcd |

### Holdout teacher: epoch selection by the original metric (2026-08-28 05:04; `rank_teacher_epochs.py` → `results/teacher_h50_epochs.tsv`)
RF-DETR-Seg-nano retrained on the 83 labelled videos (canonical rfs-01 recipe; local deviations: micro-batch 8×8, a 519-frame
proxy val for early stopping, validation every 2nd epoch, 20-epoch cap; early-stopped at epoch 18). Every epoch's EMA weights were
scored ONCE on the FULL val with the benchmark predictor (ranking pass at 10 dets) and evaluator:

| epoch | full-val segm AP50-95 | segm AP50 | bfi val F2 | val conf |
|---|---|---|---|---|
| **6 (winner)** | **0.298** | 0.604 | 0.603 | 0.23 |
| 5 | 0.294 | 0.581 | 0.585 | 0.20 |
| 8 (proxy's best) | 0.294 | 0.586 | 0.590 | 0.25 |
| 9 | 0.292 | 0.587 | 0.592 | 0.23 |
| 18 (last) | 0.291 | 0.592 | 0.593 | 0.22 |
| 4 | 0.286 | 0.608 | 0.602 | 0.23 |

The full-val ranking is flat (0.286–0.298 over epochs 4–18); the cheap proxy's best (epoch 8) is third — the two rankings
disagree on the top epoch but agree on the plateau, so the choice barely matters. The winner is then evaluated at the full
50 dets on val + test (`results/teacher_h50_{boxified,segm}_eval.tsv`; the val-selected bfi conf becomes τ of the (c) view).
Winner (epoch 6) with the benchmark tools at 50 dets (`results/teacher_h50_{boxified,segm}_eval.tsv`): box-ified bfi val F2 0.603 →
**test F2 0.577 / AP50 0.561 / AP50-95 0.247 at τ = 0.23**; segm test AP50 0.533 / AP50-95 0.210. For reference the canonical
(all-labels) teacher reaches test F2 0.697 / AP50 0.702 / AP50-95 0.328 (bfi) — half the labelled videos cost the teacher −0.12 F2,
so the h50 teacher (0.577) is still 0.19 above the sparse-half yolo26n student (0.390) it will label for.

What this teacher does on the footage nobody labelled (`holdout_teacher_on_unlabelled.py` → `results/teacher_h50_unlabelled_stats.tsv`,
τ 0.23 after ±0.25 s smoothing, exactly the labels the (c) view uses): on the 37 unlabelled NO-RIP videos it puts a pseudo box on
**3.2 %** of the native frames (1,030 / 32,169) — 0 % on 26 of the 37 videos, concentrated in four (NR-030 44 %, NR-058 30 %,
NR-008 26 %, NR-012 9 %); on the withheld expert frames of the 45 unlabelled RIP videos its boxes reach **precision 0.765 /
recall 0.611** at IoU 0.5, and 17 % of the annotated frames (1,236 / 7,112) get no pseudo box at all. So (c) trains on labels
that are ~77 % right where they exist, miss ~40 % of the expert boxes, and add a few percent of false positives on a handful of
no-rip videos.

Reference points (full labels, 3 seeds): canonical 0.483 ± 0.022, interp 0.507 ± 0.033, dense teacher 0.531 ± 0.016.

Reading so far (2026-08-27 evening): (b) − (a) = +0.007 (0.397 vs 0.390): on HALF the videos, interpolating the expert labels to
every native frame adds nothing (all three (b) seeds peak at epoch 4–6 and stop by 19–21), whereas on the full split the same
interpolation gave +0.023 (CI [−0.068, +0.101]). Densifying frames of the same videos is not the lever for yolo26n; more videos are. For yolo26s (2 seeds each) the interpolated half
DOES beat the sparse half (0.416 ± 0.015 vs 0.360 ± 0.015, +0.056 on both seeds; paired video-level +0.057 [+0.005, +0.108]) — the same capacity dependence as on the full
split (yolo26s +0.069 CI>0 vs yolo26n +0.048 CI∋0): the larger student uses denser labels, the 2.5 M one does not.
(b+) − (b) = −0.011 (0.386 ± 0.027 vs 0.397 ± 0.009): the VIDEO-LEVEL negatives of the 37 unlabelled no-rip videos add nothing on
the labelled-half setting (the labelled half already holds 37 no-rip videos as dense negatives; the full-split negatives ablation
+0.029 measured removing ALL negatives). Seed 2's +0.02 was noise. So on the unlabelled half, neither the video-level label nor
more frames of labelled videos help yolo26n; (c) tests the teacher's frame labels against (b+) = 0.386 ± 0.027 and (b) = 0.397 ± 0.009.
Video-level paired CIs (seed-pooled, B=1000, bfi): (b) − (a) = +0.006 [−0.042, +0.039] (null); (a) − full-label canonical =
−0.092 [−0.279, +0.013]; (b+) − (b) = -0.011 [-0.056, +0.074]; (b+) − (a) = -0.006 [-0.041, +0.038] (`results/student_bootstrap_cis.tsv`, GROUP_PAIRS h50_*).

**DECISIVE RESULT (2026-08-29, 3 seeds each, seed-pooled video-level bootstrap, bfi, `holdout_cis.py`):**
(c) − (b+) = **+0.072 [+0.017, +0.141]**; (c) − (a) = +0.066 [+0.001, +0.162]; (c) − (b) = +0.061 [−0.011, +0.171] (positive, NOT
resolved against interpolation alone); (c-drop) − (c) = +0.008 [−0.015, +0.030] (1 vs 3 seeds: the wrong negatives of the label-free
recipe are NOT the cost); (c) − canonical (all labels) = −0.025 [−0.224, +0.113] (unresolved at 71 test videos). Withholding half the
labels costs 0.483 → 0.390; the teacher's labels on the unlabelled half close +0.07 [0.00, +0.16] of that 0.09 (write the two
differences with their intervals — no ratio, no 'most'; the internal critic wording constraint 2026-08-29). Reading: labels on MORE VIDEOS
(even from a weaker, 0.577-F2 teacher) are what the tiny student lacks; denser labels on already-labelled videos are not.
Pre-registered rule (ICRA_2027/paper/framing_variants.md) fires → Variant B applied to main.tex.

## Per-video test F2 (2026-08-27; `pervideo_f2.py` → `results/pervideo_f2_test.tsv`, seeds micro-pooled, bfi GT, val-frozen confs)

| variant (3 seeds pooled) | rip videos: median F2 | mean | n > 0.85 | n < 0.10 (of 31) | no-rip videos with ≥ 1 FP (of 40) |
|---|---|---|---|---|---|
| yolo26n canonical | 0.246 | 0.332 | 3 | 12 | 27 |
| yolo26n dense | 0.104 | 0.322 | 4 | 15 | 25 |
| yolo26n dense + sm2 | 0.169 | 0.352 | 6 | 15 | 22 |
| yolo26s dense | 0.352 | 0.397 | 4 | 9 | 28 |

Reading (honest): the aggregate test F2 is box-weighted and dominated by the long, box-rich videos; dense supervision does
NOT raise the per-video median for yolo26n (0.25 → 0.10; the mean is flat) — the aggregate gain comes from the videos with
many rip boxes, consistent with the event-level result (no more rip VIDEOS alarmed). yolo26s dense lifts the median (0.35).
Output smoothing helps the median too (0.10 → 0.17) and reduces no-rip videos with FPs (25 → 22). Stated in Limitations.

## Label-spacing sweep (teacher-free; `build_spacing_sweep.py` + `build_spacing_views.py`; started 2026-08-27 night)

Expert anchors kept every k-th annotated frame of each rip TRAIN video (native stride 6k = 0.2·k s at 30 fps), SDF-interpolated
masks → boxes between anchors, no-rip negatives as in interp_gt_s1; yolo26n 60/15, 1 seed each (jobs esp-a2/a4/a8/a16).
References: k=1 (interp_gt_s1) 0.507 ± 0.033 (3 seeds); sparse-only 0.483 ± 0.022.

| anchors every | of the expert masks | spacing (s @30 fps) | train images | best / stopped | test F2 (bfi) | AP50 | AP50-95 | F2 (tight) | job |
|---|---|---|---|---|---|---|---|---|---|
| 2nd frame | 50 % | 0.4 | 201,691 | 2 / 17 | 0.470 | 0.472 | 0.223 | 0.470 | 3181764 esp-a2 |
| 4th frame | 25 % | 0.8 | 201,162 | 3 / 18 | 0.520 | 0.502 | 0.230 | 0.516 | 3181765 esp-a4 |
| 8th frame | 13 % | 1.6 | 200,052 | 2 / 17 | 0.474 | 0.479 | 0.170 | 0.487 | 3181910 esp-a8 |
| 16th frame | 6.5 % | 3.2 | 197,718 | 3 / 18 | 0.432 | 0.413 | 0.178 | 0.450 | 3182011 esp-a16 |

Reading (single seeds; the full-interpolation seeds span 0.47–0.55, so differences < 0.05 are noise): from 100 % down to 13 % of
the expert masks (anchors every 0.2–1.6 s) the interpolated view stays in the 0.47–0.52 band around sparse-only (0.483) and full
interpolation (0.507); at 6.5 % (one anchor every 3.2 s) it falls to 0.432, below sparse labelling. Practical guideline: SDF
interpolation tolerates an expert annotation every ~1.5 s of video; beyond ~3 s the interpolated boxes hurt more than they help.
Order is not monotone (a4 > a1 > a8 ≈ a2) — consistent with noise, not with a density effect for yolo26n (cf. holdout (b) ≈ (a)).

## Causal output smoothing (2026-08-29 13:51; `smooth_student_outputs.py` CAUSAL=True, window [t−2k, t], NMS 0.7 re-applied; rows *_csm1/_csm2)

Same dumps, same evaluator
(val-swept conf per dump, frozen, test). k=1 stays the a-priori primary (pre-registered 2026-08-27); k=2 sensitivity. bfi / tight, 3 seeds each:

| student | raw | ±1 symmetric (sm1) | causal k=1 (csm1) | causal k=2 (csm2) |
|---|---|---|---|---|
| canonical yolo26n | 0.483±0.022 / 0.500±0.026 | 0.507±0.025 / 0.527±0.033 | 0.507±0.026 / 0.529±0.033 | 0.511±0.026 / 0.533±0.032 |
| dense yolo26n | 0.531±0.016 / 0.532±0.008 | 0.557±0.017 / 0.561±0.009 | 0.557±0.017 / 0.559±0.007 | 0.560±0.017 / 0.565±0.009 |
| dense yolo26s | 0.550±0.013 / 0.557±0.018 | 0.570±0.016 / 0.575±0.020 | 0.572±0.016 / 0.578±0.021 | 0.577±0.017 / 0.582±0.024 |

Reading: the smoothing needs no look-ahead — a causal window over the past two/four sampled frames gives the same +0.02–0.03 as the
symmetric ±1 window (differences ≤ 0.005, far inside the seed spread), so the pipeline can apply it online at no output delay. VAL
prefers csm2 over csm1 by ~0.002 on every seed (not a meaningful margin; csm2 reported as sensitivity only, no new test selection).

## i.MX95 prep: int8 TFLite exports + local sanity (2026-09-01 11:51; export_tflite_imx.py / validate_tflite_imx.py, env icra_tflite; artefacts /mnt/linux/icra_edge/exports/tflite_imx/)

The i.MX95 Pro's Neutron NPU (eIQ) takes int8 TFLite, not ONNX. Exported (ultralytics 8.4.92 int8): t1_n_s1 (dense yolo26n),
canon_n_s1, gate_cls224 (= benchmark 3_cls_yolo26_nano_224). Local SANITY on 200 val frames vs the PyTorch dumps at the same
conf (NOT benchmark rows — the board rows come from eIQ on the device; results/tflite_sanity.tsv):

| artefact | F2 tflite | F2 dump | gap | verdict (tolerance 0.03) |
|---|---|---|---|---|
| t1_n_s1_int8.tflite | 0.8245 | 0.8155 | +0.009 | PASS |
| canon_n_s1_int8.tflite | 0.8977 | 0.8739 | +0.024 | PASS |
| gate_cls224_int8.tflite | decision agreement vs the .pt gate at 0.50: 200/200 = 1.000 | | | PASS |

(F2 here is on the 200-frame subset (mostly rip-dense early val videos), far above the full-split numbers by construction —
only the GAP is the check.) Gotchas hit: ultralytics leaves the exported tflite NEXT TO the checkpoint (moved out of the
benchmark best_models dir); os.replace fails cross-device (shutil.move); COCO file_name is relative to images/; a tflite
without task metadata loads as detect — pass task="classify" for the gate.

## Device week: first measured ladders (2026-09-03 20:45; provenance = branch device-week, ICRA_2027/device_results/{board_latency,jetson_trtexec_latency}.tsv; measured by the ICRA2027-device session/Codex on the test PC)

**RPi 4 Model B Rev 1.1 (heatsinks, no fan; Debian 13, onnxruntime CPU, 4 threads; board_bench v2 with thermal columns):**
t1 dense yolo26n 384/512/640: infer 137.4/242.7/401.9 ms (6.8/3.8/2.4 fps e2e); canonical n identical (same arch:
139.0/248.3/397.1); dense yolo26s 435.6/760.0/1190.1 ms; cls gate@224 31.1 ms = 29.3 fps. Gate is ~4.4x cheaper than the
smallest detector rung -> 1 Hz gating costs ~3% of one core-second per second. Temps drifted 49->80 C across the ladder
(recorded per row); the first ladder attempt (Rev 1.2 board, no heatsinks) was invalidated by throttling at 83 C and the
original SD card died - board swapped to Rev 1.1 + heatsinks, fresh OS, all provenance in NOTES.md.

**Jetson Nano DK 2019 (L4T R32.7.1, TensorRT 8.2, MAXN, clocks on; engines from the _trt82 Mod-rewrite graphs - parse fix
verified on device):** t1 dense n FP16 384/512/640: 18.6/28.5/45.0 ms trtexec latency (52.6/34.0/21.8 qps); canonical
identical. Gate FP16: 4.17 ms (239 qps). The 2019 Nano sustains ~8x the RPi 4's CPU rate at 384.

Pending from the boards: Shelly-logged power windows (+ ZX cross-check), RPi accuracy dumps, full-system camera rows;
Orin Nano Super arrives 09-03, measured 09-04.

## Device power rows (2026-09-04 20:32; Shelly Plug M Gen3 protocol windows, 10 readings each; ICRA_2027/device_results/device_wall_meter.tsv)

| board | idle W | t1@384 | t1@640 | canon@640 | s_s3@384 | gate@224 |
|---|---|---|---|---|---|---|
| RPi 4 (bare, no fan) | 2.61 | 6.56 | 5.99 | 5.95 | 6.12 | 6.05 |
| RPi 5 (fan) | 3.08 | 13.62 | 12.73 | 12.76 | 14.32 | 14.24 |
| RPi 4 (heatsinks, no fan) | 2.440 | 6.510 | 6.280 | 6.350 | 6.420 | 6.120 |
| Jetson Nano 2019 (MAXN, FP16 TRT) | 3.02 | 9.29 | 10.25 | 10.59 (640) | — | — |

Latency (RPi 5 fan, no throttling): t1 38.8/74.4/118.6 ms (384/512/640; 24.1 fps e2e at 384), s_s3 113.5/212.7/334.8, gate 7.5 ms
(117 fps). Energy per e2e decision, t1@384: RPi 5 ~565 mJ gross / ~437 net vs RPi 4 ~976 / ~586 -> the Pi 5 draws twice the power
but is 3.5x faster and WINS on energy per decision. Both boards' dumps are bit-identical (onnxruntime CPU determinism across
aarch64 boards). Device accuracy at frozen confs (rescaled from copy space - the coordinate bug was home-side, fixed in
score_device_dumps.py; results/device_eval_{bfi,tight}.tsv): t1@640 0.542/0.540 (bfi/tight), t1@384 0.545/0.545, canon@640
0.529/0.534, s_s3@384 0.560/0.567. These sit ABOVE the full-res 4090 rows (+0.03) - image-pipeline effect (INTER_AREA+JPEG copies);
the same-copies workstation reference (unit icra-wscopy) isolates it; deployment gate per the pre-declaration applies to
device-vs-same-copies only.

### Deployment-gate decomposition (2026-09-04 20:46; scorer results/device_eval_{bfi,tight}.tsv, dumps in ICRA_2027/device_results/)

Three-way chain on the same 640px copies at frozen confs: DEVICE (RPi, onnxruntime) vs X86ONNX (identical board_bench code
path on x86) vs WS-PYTORCH (ultralytics on the 4090). Result: device - x86 is |d| <= 0.0002 F2 on all 8 (4 configs x 2
views) - the boards execute the deployed graph essentially exactly (aarch64/x86 SIMD drift flips <= 12 of ~25k preds at the
0.001 floor, none at the operating points). Device - ws-pytorch: t1@384 +0.021/+0.016, t1@640 +0.006/+0.006, canon -0.001/
+0.003, s_s3 +0.004/+0.004 -> the whole gate discrepancy is EXPORT-TIME (square-input letterbox + in-graph e2e head vs
PyTorch minimal-rect + framework NMS), favorable in direction, and NOT device compute or the image copies. Gate verdicts
as pre-declared (ws-pytorch reference): canon + s_s3 PASS, t1 favorable-fail - reported with this decomposition; lesson
for the paper: declare fidelity gates against the DEPLOYED-graph reference.

**Energy per decision, t1@384 gross (2026-09-05 10:34):** Jetson Nano 176.6 mJ (net 119.2) << RPi 5 ~565 << RPi 4 ~976 — the 2019
GPU tier is ~5.5x more energy-efficient per decision than the RPi 4 CPU and ~3.2x than the RPi 5, at 7.4x / 3x their
inference speed. GPU offload wins the energy race even on 2019 silicon; the Orin numbers will extend the curve.

## Orin Nano Super first ladder (2026-09-05 13:16; NOTES 2026-09-05 14:10 + jetson_trtexec_latency.tsv; JetPack 6.1 / L4T R36.4.2, TRT 10.3, MAXN + jetson_clocks, SD boot)

FP16, original ONNX graphs (no TRT-8.2 rewrite needed): t1 2.44/3.10/4.31 ms at 384/512/640 (427/345/251 qps); canonical
identical. Cross-generation strip at t1@384 inference: RPi 4 CPU 137.9 ms -> Nano 2019 18.6 ms -> Orin Nano Super 2.44 ms
(56x / 7.6x). tegrastats rails readable (VDD_IN, VDD_CPU_GPU_CV, VDD_SOC); temps 53-56 C during runs. Pending on-device:
INT8 builds (calib_640), power modes 7/15 W, protocol power windows (Shelly + rails), accuracy dumps, camera rows.

**Orin power windows (2026-09-05 14:42; MAXN + pinned clocks, Shelly):** idle 8.18 W (clocks pinned!), t1@384 load 15.69 W ->
**36.6 mJ/decision gross (17.5 net)**. Energy-per-decision ladder at t1@384 gross: RPi 4 976 -> RPi 5 565 -> Nano 2019
176.6 -> Orin Nano Super 36.6 mJ (27x RPi 4). Caveat for the budget: the 8.2 W pinned idle dominates gated-duty
deployment cost at MAXN; the 7 W / 15 W modes + dynamic clocks rows are the deployment-relevant ones - pending.

## Copy-pipeline resolution surprise — replicated, under val-only investigation (2026-09-06 09:22)

Orin dumps added canon@384/512 (never dumped on-device before): **canon@384 F2 0.604 bfi** vs canon@640 0.529 and t1@384
0.545 — the canonical model gains +0.075 at 384 px ON THE 640px COPIES. Replicated independently (x86 same code path:
0.6043/0.5951; Orin TRT FP16/FP32 within 0.001 of x86 everywhere; t1 and s_s3 rows consistent across all engines).
NOT an engine or threshold artifact (AP50 jumps equally: 0.510→0.568). Hypothesis: the copies interpose an INTER_AREA
downscale to 640 before the 384 letterbox (double resize with antialiasing) — absent in the 4090 full-res protocol
(canonical was flat 640→384 there) and absent in a deployment camera path. If true, "INTER_AREA pre-downscale to ~1.7x
target, then letterbox" is a FREE preprocessing gain. VAL-ONLY check running (val_resize_pipeline_check.py, pre-registered
in TEST_CONSUMPTION: direct-384 vs area-640-then-384, canonical + t1, val sweep, no test read). NO paper claim until it
lands; the device ladder table cites the copy-pipeline rows as measured with the pipeline named.

**Val verdict on the double-resize (2026-09-06 09:37; val_resize_pipeline.tsv, VAL only):** INTER_AREA pre-downscale to 640 before the
384 letterbox is a REAL, model-agnostic gain on full-res frames: canonical 0.4837 -> 0.5094 (+0.026), t1 0.4880 -> 0.5052
(+0.017), val-swept confs. It explains ~a third of the copy-pipeline surprise; the remaining canonical-specific ~+0.05 on
copies stays a reported-with-pipeline footnote (entangled with square-vs-rect letterbox and the test distribution - no
claim). Deployment recipe gains one line: pre-downscale the camera frame with INTER_AREA to ~1.7x the input size before
letterboxing (free, +0.02 F2 class). Model choice unchanged (val B-pipeline: canonical 0.509 ~ t1 0.505; t1 stays the
pre-declared deployment model).

## Orin power-mode axis (2026-09-06 12:48; jetson_orin_power_mode_{summary,power}.tsv; t1@384 FP16, jetson_clocks on, python TRT loop with JPEG decode)

| nvpmodel | GPU infer ms | e2e ms | fps (with JPEG) | idle W | load W | mJ/decision gross | net |
|---|---|---|---|---|---|---|---|
| 7 W | 7.81 | 8.36 | 49.4 | 5.38 | 6.60 | 141 | 26 |
| 15 W | 4.27 | 4.72 | 80.2 | 6.01 | 8.00 | 102 | 25 |
| MAXN (25 W) | 2.78 | 3.21 | 104.9 | 6.70 | 9.51 | 92 | 27 |

Reading: the NET energy per decision is mode-invariant (~26 mJ) - the power mode buys speed and pays idle; gross mJ/decision
falls with speed because idle is amortised over more decisions. For a gated deployment at <=1 Hz the mode is irrelevant
to energy (idle-dominated, 5.0 W with dynamic clocks per the duty rows) and the 7 W mode's lower ceiling is free insurance
against thermal/PSU limits. (The python-loop e2e numbers include JPEG decode; trtexec GPU-only at MAXN was 2.44 ms.)

## Orin explicit-Q/DQ INT8 — first scored rows (2026-09-06 15:25; engines from the v3 graphs, TRT 10.3 --int8 --fp16; device_eval TSVs)

| config | INT8 F2 bfi / tight | FP16 F2 bfi / tight | delta | P / R (bfi) INT8 vs FP16 |
|---|---|---|---|---|
| t1@384 | 0.497 / 0.496 | 0.545 / 0.545 | −0.048 | 0.523/0.491 vs 0.485/0.562 |
| t1@512 | 0.517 / 0.516 | 0.544 / 0.546 | −0.028 | 0.636/0.493 vs 0.512/0.553 |

Reading: INT8 at the FP32-frozen conf is more conservative (precision up, recall down) - a score-distribution shift, i.e.
partly threshold miscalibration rather than pure capacity loss; AP50 also drops (0.485->0.453 at 384) so part is real.
Protocol: conf stays frozen (a val re-sweep for INT8 would be legitimate val-only selection but needs board VAL dumps -
not planned before freeze; stated as a caveat). Remaining INT8 dumps (t1@640, canonical x3) + latency/power pending.

## INT8 verdict on the Orin Nano Super (2026-09-06 15:36; all six v3 Q/DQ engines: latency.tsv + scored dumps) — NOT WORTH IT

| config | INT8 ms (qps) | FP16 ms (qps) | INT8 F2 bfi | FP16 F2 bfi | dF2 |
|---|---|---|---|---|---|
| t1@384 | 2.74 (378) | 2.44 (427) | 0.497 | 0.545 | −0.048 |
| t1@512 | 3.45 (306) | 3.10 (345) | 0.517 | 0.544 | −0.027 |
| t1@640 | 4.49 (237) | 4.31 (251) | 0.497 | 0.542 | −0.045 |
| canon@384 | 2.73 (379) | 2.44 (427) | 0.419 | 0.604 | −0.185 |
| canon@512 | 3.45 (306) | 3.10 (345) | 0.255 | 0.594 | −0.339 |
| canon@640 | 4.56 (234) | 4.30 (252) | 0.454 | 0.529 | −0.075 |

Explicit INT8 is SLOWER than FP16 on this board for these 2.5 M-parameter graphs (+8-12% latency: Q/DQ overhead with
FP16 fallback layers; FP16 tensor cores already saturate at batch 1) AND less accurate at the frozen operating points
(t1 −0.03 to −0.05; the sparse-trained canonical collapses, recall 0.21 at 512). Energy per decision is therefore worse
by construction -> the planned INT8 power windows are cancelled. Honest negative result: FP16 is the deployment
precision; INT8 rows go in the ladder table with this sentence. (Caveat kept: confs frozen from FP32 val; a val
re-sweep for INT8 would recover some of the threshold shift but not the AP50 loss.)

## Orin matched power matrix, 18 windows (2026-09-06 19:52; orin_qdq_v3/power/power.tsv; engine-only, model-loaded idle per mode)

MAXN INT8 (idle 8.62 W): t1 38.0/51.8/72.5 mJ gross (15.1/23.6/36.1 net) at 384/512/640 - vs FP16 MAXN 36.6/52.0/78.1.
7 W FP16 (idle 5.97 W): t1 8.19/8.74/9.07 W load, 140.6/109.0/74.0 qps -> 58.3/80.2/122.6 mJ gross (15.8/25.4/41.9 net).
7 W INT8: 7.86/8.12/8.34 W, 129/96/69 qps -> 60.9/84.8/120.9 gross (14.7/22.4/34.3 net). Canonical identical within noise.
Reading: INT8 draws ~4% less but runs ~8% slower -> gross mJ/decision WORSE than FP16; net within noise. Together with
the latency and accuracy rows, INT8 is closed as a fully measured negative. Deployment reference row (paper): 7 W FP16
t1@384 = 8.19 W, 141 qps, 58 mJ gross / 16 mJ net per decision. Paper: tab:device filled with measured rows (F2 tight,
GPU ms, fps, mJ MAXN/7 W) + ladder paragraph rewritten; builds 7 pp.

## Deployed configuration on the board's own outputs (2026-09-06 20:56; deployed_cascade_metrics.py -> results/deployed_cascade_test.tsv, deployment_gates.tsv)

Declared config (gate@224 thr .50 through the DEPLOYED preprocessing = board_bench.letterbox, x86 same code path; Orin FP16
t1@384 dump; conf .04; 1 Hz; k=n=1; GAP 10 s), video-level bootstrap B=1000:
- gated 1 Hz: rip videos alarmed 0.516 [0.355, 0.710]; TTA median 0.0 / p90 4.0 s; no-rip false-episode clips 0.025 [0.000,
  0.075]; detector runs on 7.4% of frames. Ungated 1 Hz: 0.742 [0.581, 0.903] / 0.175 [0.075, 0.300] / 20.6%.
- gated 0.2 Hz: 0.484 [0.290, 0.645]; false-episode 0.025; detector 1.6%. gated 5 Hz: 0.581; 0.050; 35%.
- sparse-label baseline (canon@384, same board): gated 1 Hz 0.548 / 0.050; ungated 0.677 / 0.225.
Gates: G1 PASS (0.516 vs 0.548), G2 PASS (0.025), G3 PASS (0/4.0 s), G4 latency PASS + accuracy FAIL-upward (declared
reference), G5 FAIL (Orin idle floor, ~4 h on 20 Wh). KEY FINDING: the gate through the deployed preprocessing agrees with
the offline (ultralytics-preprocessed) gate on only 84.2% of val frames (frame recall 0.632, no-rip pass 0.829 at .50), and is
HARSHER: on the board the cascade trades recall 0.74->0.52 for false-alarm clips 0.18->0.03, vs the offline replay's 0.35->0.10
for 4-8 recall points. Abstract/contributions now carry the deployed numbers; the offline replay stays as the analysis.
Smoothing at deployed cadence (VAL, results/smoothing_at_cadence_val.tsv): 1 Hz +0.027/+0.030/+0.017, 0.2 Hz +0.037/+0.036/+0.027
-> helps at every cadence, every seed; deployed row keeps smoothing ON.

## Deployed configuration, CORRECTED gate path (2026-09-07 06:29; gate = training transform; deployed_cascade_test_centercrop.tsv, deployment_gates_centercrop.tsv)

gated 1 Hz: rip videos alarmed 0.613 [0.452, 0.774]; TTA 0.0/1.4 s; no-rip false-episode 0.100 [0.025, 0.200]; detector 9.5 %.
ungated 1 Hz: 0.742 [0.581, 0.903] / 0.175 [0.075, 0.300]. gated 0.2 Hz: 0.581 / 0.075 / 2.0 %. gated 5 Hz: 0.677 / 0.125 / 45 %.
Sparse-label baseline gated 1 Hz: 0.613 / 0.125. Gates: G1-G3 PASS, G4 accuracy upward-fail, G5 FAIL (unchanged).
v1 (letterbox) rows: 0.516 / 0.025 - archived (*_letterbox), reported only as the preprocessing lesson. Paper updated.

## Board-fitted detector on the Orin (2026-09-07 06:34; pre-declaration 3; deployed_cascade_test_centercrop.tsv)

Gate .50 (training transform) + dense yolo26s@384 (conf .01, bfi-frozen) at 1 Hz: rip videos alarmed 0.613 [0.419, 0.774]
(= nano's 0.613), no-rip false-episode 0.175 [0.075, 0.300] (nano 0.100); ungated 0.742 / 0.375 (nano 0.742 / 0.175).
Reading: at its F2-optimal (recall-leaning, conf .01) operating point yolo26s is NOT a better alarm than nano - same
video recall, twice the false-alarm clips. Frame-level F2 ordering != alarm-level ordering. The gate makes a bigger
detector affordable, but "best model that fits" must be judged at the alarm level; an alarm-level val-selected conf for
yolo26s was not done before the freeze (stated).

## Scenario S2: classification-only lifeguard alert (2026-09-07 06:53; declared before read)
Val (1 Hz, no detector, video-F2 selection): yolo26n-224 thr 0.9 -> 0.867 recall / 0.211 false-episode; resnet34-640 thr 0.9
0.867 / 0.184 (val-equal; feasibility only); yolo26m/x reject (false-episode >= 0.45). TEST (deployed-path probs, thr 0.9):
rip videos alarmed 0.742 [0.581, 0.871], no-rip false-episode clips 0.225 [0.100, 0.350], TTA 0/0 s — vs the cascade's
0.613 / 0.100. Reading: with a lifeguard on duty, the classifier alone alerts on more rip videos (+0.13) at 2.3x the false
alerts; the cascade is the unattended mode. Cost: the gate alone (0.84 ms Orin / 31 ms RPi 4 / 7.5 ms RPi 5) at 1 Hz = the
board's idle floor. Classify-only camera windows added to the device list (capture_pipeline_power CLASSIFY_ONLY).

**Correction (2026-09-07 07:07, device review item 4):** the INT8 energy statements above ("worse by construction", "NOT WORTH IT on
energy") are over-general. Audited 7 W pairs: gross −1.4 % to +6.5 % (higher in 5 of 6; t1@640 lower), net −18 % to −7 %
(lower in all 6). Correct wording: INT8 is slower and less accurate; its energy sign depends on the accounting boundary.

## Per-board deployed configurations, declared rule (fits = pipeline <= 1 s; best = val video-F2 winner) — 2026-09-07 08:04
Orin FP16 dumps scored: expert yolo26s 384/640 F2 tight 0.497/0.549 (= x86 device-equivalent), RF-DETR-n@384 **0.706** (AP50 0.705;
benchmark full-res 0.695). Cascade reads (gate .50 training transform, 1 Hz, frame-frozen confs, one read each):
| detector (Orin) | rip alarmed | false-episode clips | ungated |
| RF-DETR-n (val winner, deployed) | 0.742 [0.581, 0.871] | 0.075 [0.000, 0.150] | 0.871 / 0.425 |
| dense yolo26n (pre-declared, gates) | 0.613 [0.452, 0.774] | 0.100 [0.025, 0.200] | 0.742 / 0.175 |
| dense yolo26s | 0.613 [0.419, 0.774] | 0.175 [0.075, 0.300] | 0.742 / 0.375 |
| expert yolo26s | 0.677 [0.484, 0.839] | 0.175 [0.050, 0.300] | 0.774 / 0.500 |
Val ranking (RF-DETR-n > expert s > dense s > nano) is reproduced on test at the alarm level for the top and bottom. RPi 5: RF-DETR-n
fits (394 ms) -> its dump/read pending; RPi 4: pending latency decides.

**RPi 5 deployed config (2026-09-07 11:09):** RF-DETR-n (ORT CPU, 394 ms, fits) behind the gate at 1 Hz: 0.742 [0.581, 0.903] / 0.075
[0.000, 0.175] — identical to the Orin (RF-DETR-n CPU dump F2 0.7060 vs TRT FP16 0.7056: transformer cross-engine fidelity
too). Energy price on the Pi 5: 6.4 J/decision gross (vs nano 0.57 J) -> ~0.6 W above idle at 10 % duty. Expert-s@384
rpi5 0.522/0.496 (= Orin/x86). RPi 4: pending (RF-DETR-n latency decides whether it fits).

**Nano 2019, expert yolo26s@384 FP16 via the Mod rewrite (2026-09-07 11:41):** F2 0.522 bfi / 0.495 tight — matches Orin TRT, RPi 5 ORT and
x86 (fourth independent engine). Pipeline 52.6 ms, trtexec 40.5 ms. RF-DETR-n does not parse on TRT 8.2 (no LayerNormalization
plugin) - documented feasibility boundary. Nano power deferred (meter moved); webcam rows tomorrow.

## 2026-09-12 01:28 — Cascade v2 (threshold correction + extended ladder), verified against the test-PC audit
Correction: v1 of deployed_cascade_metrics.py scored expert yolo26s and dense yolo26s at their BFI frame confs (.05/.01); the cascade
tables are on the tight view, whose frozen confs are .07/.02 (alarm_op_rule_val.tsv). Caught by the test-PC agent
(device_results/home_takeover_20260909/PAPER_CORRECTIONS.md #2). The paper's "dense-small twice the false clips" sentence rested on
the wrong threshold and is withdrawn. v2 = all rows at tight confs, RF-DETR rows from the INTER_AREA re-dumps, extended rows added.
Every row below reproduces device_results/home_takeover_20260909/corrected_cascades.tsv (independent scorer, same counts).
gate .50 @1 Hz, k=n=1 (rip videos alarmed / no-rip clips with a false alarm; video bootstrap B=1000 seed 0):
- Orin dense-n .04: 0.613 [0.45,0.77] / 0.100 [0.03,0.20]  (unchanged; gates table stands)
- Orin canonical-n .03: 0.613 / 0.125 (unchanged)
- Orin dense-s .02: 0.613 [0.42,0.77] / 0.125 [0.03,0.23]   (v1 at .01: 0.61 / 0.20 -> superseded)
- Orin expert-s .07: 0.645 [0.45,0.81] / 0.125 [0.03,0.23]  (v1 at .05 superseded)
- Orin RF-DETR-n .23 (INTER_AREA): 0.742 [0.58,0.90] / 0.075 [0.00,0.18]; ungated 0.871 / 0.375  (INTER_LINEAR dump gave the same counts)
- Orin RF-DETR-s .26: 0.742 / 0.125; RF-DETR-m .31: 0.742 / 0.100 (ungated 0.903 / 0.275)
- RPi 5 RF-DETR-n: 0.742 / 0.075 (identical to Orin); RPi 5 RF-DETR-s: 0.742 / 0.125
- RPi 4 (both units) expert-s .07: 0.645 [0.48,0.81] / 0.125 [0.05,0.23]; Jetson Nano expert-s (TRT 8.2 FP16): 0.645 / 0.150
- i.MX95 CPU FP32 dense-n: 0.613 / 0.100 (= Orin dense-n); i.MX95 NPU QAT2-adapted dense-n (native images): 0.613 [0.42,0.77] / 0.125 [0.05,0.25]; ungated 0.710 / 0.275
Provenance: results/deployed_cascade_test_centercrop.tsv (v2), DUMPS block of deployed_cascade_metrics.py; log /mnt/linux/icra_edge/logs/cascade_v2.log.
Frame-level device rows (test-PC scorer, tight view, frozen confs; device_results/home_takeover_20260909/device_accuracy.tsv):
RF-DETR-n 0.7026 (Pi4/Pi5 CPU, bit-identical) / 0.7024 (Orin FP16); RF-DETR-s 0.7141 (Pi5) / 0.7156 (Orin); RF-DETR-m 0.6739 (Orin);
expert-s@384 0.4961 (Pi4/Pi5) / 0.4970 (Orin) / 0.4953 (Nano); expert-s@640 Orin 0.5486; dense-n@384 i.MX95 CPU 0.5449 (= Orin 0.545).
i.MX95 QAT2 native TEST (nxp_selected_test_20260910/metrics.json): tight F2 0.5033 vs FP32 reference 0.4981 on the same native
letterbox-384 tensors, paired video bootstrap +0.0052 [-0.0044, +0.0147]; AP50 0.463/0.463. Adapted model (TRAIN-only QAT, VAL-selected
among four predeclared variants) - NOT an equivalent export; reported as documented feasibility.

## 2026-09-12 — device-week ingestion (Codex branch merged at 8252841), cascade v2, camera windows, i.MX95 NPU

**Cascade v2 (results/deployed_cascade_test_centercrop.tsv, script v2):** v1 replayed expert yolo26s and dense yolo26s at their
BFI-view confs (.05/.01) although the cross-detector tables are on the tight view (.07/.02) — caught by Codex's home-takeover
audit (device_results/home_takeover_20260909/PAPER_CORRECTIONS.md §2). Corrected, gate .50 @1 Hz (rip videos alarmed / no-rip
false clips, video bootstrap B=1000 seed 0): dense-n Orin 19/31 = 0.61 [0.45,0.77] / 4/40 = 0.10 [0.03,0.20] (unchanged);
dense-s 19/31 / 5/40 = 0.13 (v1 said 8/40 → "twice the false clips" withdrawn); expert-s 20/31 = 0.65 [0.48,0.81] / 5/40 on
Orin, RPi 4 bare, RPi 4 heatsink (bit-identical dumps) and 6/40 on the 2019 Nano (TRT 8.2 FP16); RF-DETR-n (INTER_AREA
re-dumps, rf_area_20260908 + rpi5_overnight_20260908) 23/31 = 0.74 [0.58,0.90] / 3/40 = 0.08 [0.00,0.18] on Orin and RPi 5
(same counts as the withdrawn INTER_LINEAR read; no gate 27/31, 15/40 = 0.38); RF-DETR-s 23/31 / 5/40 (Orin, RPi 5); RF-DETR-m
23/31 / 4/40 (Orin); i.MX95 CPU FP32 dense-n 19/31 / 4/40; i.MX95 NPU QAT2 (native dump) 19/31 = 0.61 [0.42,0.77] / 5/40 = 0.13
[0.05,0.25], no gate 22/31 / 11/40. Every shared cell equals Codex's independent scorer (corrected_cascades.tsv).
**Frame F2 (tight, device dumps, Codex device_accuracy.tsv):** RF-n 0.7026 (Pi4/Pi5) / 0.7024 (Orin) — the INTER_LINEAR device
row was 0.706; RF-s 0.7141 (Pi5) / 0.7156 (Orin); RF-m 0.6739 (Orin); expert-s@384 0.4961 (Pis) / 0.4970 (Orin) / 0.4953 (Nano);
expert-s@640 Orin 0.5486; i.MX95 CPU dense-n 0.5449.
**Per-board rule outcome (tab:perboard):** Orin → RF-n (44 ms cached pipeline at 7 W dynamic, 41.8–50.2 ms over 5 repeats;
7 ms = engine only); RPi 5 → RF-n (400 ms; RF-s fits at 818 ms but val .845 < .872); RPi 4 → RF-n does NOT fit (1646/1754 ms
heatsink/bare) → expert yolo26s (411/424 ms); Nano → expert yolo26s (52.6 ms; RF-n unsupported on TRT 8.2); i.MX95 → QAT2 NPU
(46.0 ms pipeline / 34.4 ms NPU, FRDM board). Sources: home_takeover_20260909/integrated_measurements.tsv,
imx95_frdm_measurements_20260912/cases.
**Power (Shelly only; the PI withdrew the revolt cross-check 2026-09-11):** Orin RF-n 7 W: idle 5.23 / load 8.34 W, 21.9 qps →
381 gross / 142 net mJ; RPi 5 RF-n: 3.38 / 15.83 W → 6341 / 4987 mJ; RPi 4 heatsink expert-s 2.87 / 6.66 W → 2850 / 1622 mJ; bare
2.92 / 6.33 W; Nano expert-s MAXN 3.59 / 10.37 W → 429 / 280 mJ; i.MX95 FRDM QAT2 NPU 4.77 / 6.81 W → 313 / 94 mJ; FRDM NPU gate
4.77 / 6.95 W, 85 qps → 82 / 26 mJ.
**Whole-unit camera windows (results/camera_policy_rows.tsv ← make_camera_policy_table.py; Codex completed_board_review +
Pi4 result.json):** 600 s, USB camera at a dark-room monitor replay, four policies; gated no-rip W: Orin 6.72, Nano 5.61,
i.MX95 Pro 9.58, RPi 5 5.62, RPi 4 heatsink 5.21 (bare pending); gating saves 44 % on the RPi 5 no-rip window and ~0 on the
Jetsons/NXP (idle-dominated). Max-throughput decisions/s: Orin 18.6 (MAXN), NXP 20.3, Nano 7.7, Pi4 4.4, Pi5 1.8.
**Deployment budget (deployment_budget.py from the gated no-rip rows):** Orin 161 Wh/d → 58 W panel, 504 Wh LFP, ideal BOM
~465 EUR; RPi 4 heatsink 125 Wh/d → 45 W, 391 Wh, budget BOM ~196 EUR; RPi 5 135 Wh/d.
**i.MX95:** original int8 TFLite export (2026-09-01) was calibrated without a RipBench set (ultralytics fallback) → clipped gate
probabilities, golden fail; Codex re-calibrated the gate (passes golden on the NPU) and, for the detector, trained a TRAIN-only
QAT backbone adaptation (4 predeclared variants, VAL-selected QAT2, native VAL tight F2 .4924 vs FP32 .4964), one sealed native
TEST read .5033 vs .4981 (paired +.0052 [-.0044,+.0147]). Reported as a feasibility row with an adapted model, not fidelity.
**HIL v2 (deployment_repair_20260907/orin_7W/…RFcascade_none):** 70 ticks, 30 detector calls, 0 missed deadlines, 1 input-
unavailable tick; fault runs (disconnect/stale/overload) report unknown/stale, never a silent negative. The v1 letterbox HIL row
(45 ms lateness, 4.84 W rails) is withdrawn from the paper.
**Paper:** all ten PAPER_CORRECTIONS items applied 2026-09-12 (revolt removed; per-row clock policy; 128-frame Q/DQ calibration;
fidelity claim scoped to the four predeclared configs + per-model export gates; chronology of the fits/best rule; 44 ms not 7 ms;
0.38 ungated; dense-s 0.13 vs 0.10; tab:perboard filled; tab:camera added; budget from camera rows; i.MX95 paragraph).

**2026-09-12 02:00 — tiered-lever paragraph (the PI):** Orin = gate buys accuracy (0.18→0.10 student; 0.38→0.08 RF-n), energy = platform floor; RPi 5 = gate buys 44 % of no-rip unit power (10.10→5.62 W); RPi 4/Nano tier = only small students fit at 1 Hz (RF-n 1.65/1.75 s), label density is the lever (+0.07 frame F2 yolo26s dense 0.568 vs expert 0.496 @384 device pipeline; alarm-level tie 19/31 vs 20/31, 5/40 both); RF-n at 0.2 Hz behind the gate on the RPi 4 = 21/31, 3/40 (Codex row02/row04, descriptive). Contribution (iii) carries the same tiering.

**2026-09-12 03:02 — Codex-critic corrections applied (paper_edits/2026-09-12_*):** constant "≈20 mJ per decision at every duty" WITHDRAWN — net
mJ from jetson_orin_duty_power.tsv is 20.0 / 20.0 / −30 / −298 at 5 / 2 / 1 / 0.2 Hz (load below idle at ≤1 Hz), so only "gross draw near
the platform floor" is claimed; expert-s export gate −0.008 (recompute .5064 vs .4986), not −0.010; "never a silent negative" withdrawn
(9/9 recheck fixtures emit VALID on loss during inference); RF-n camera runtime uses UNSMOOTHED detections (C36) — cadence smoothing gains
are dense-yolo26n VAL only; holdout/spacing/ablation sentences hedged (no equivalence, no "lower bound", one-seed sweep descriptive);
Limitations replaced; energy budget = illustrative (prices exclude board+camera, no enclosure-temperature claim); AI-use disclosure added
(CFP requirement). Codex-test: Pi 4 camera windows ran the dense yolo26n (not expert-s) — labels/caption/CLAIMS corrected.
