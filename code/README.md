# code/ — the pipeline scripts, one per stage the paper reports

Scripts are constants-at-top with no CLI arguments: edit the path constants for your layout (they name the original machines' paths:
`/home/user/RipBench/...` = dataset and frame caches, `/mnt/linux/icra_edge/...` = bulky outputs not included here,
`ICRA_2027/device_results/` = the per-window evidence tree (not included here), `results/` = `results/`, `manifests/` = `data/`).
The folder is flat because `bootstrap_cis.py`, `board_bench.py` and `build_teacher_view.py` are imported by sibling scripts.

| Stage | Scripts | Output here |
|---|---|---|
| Teacher labels and label views | `teacher_predict_frames.py`, `build_teacher_view.py` (teacher view with temporal smoothing), `build_interp_view.py` (interpolated expert view), `build_ablation_views.py`, `build_spacing_sweep.py` | `label_views.tar.gz` asset |
| Video-level holdout | `build_holdout_split.py`, `build_holdout_views.py`, `teacher_retrain_h50.py`, `build_h50_teacher_view.py`, `holdout_cis.py` | `data/holdout_h50_seed0.tsv`, `results/teacher_h50_*.tsv` |
| Student training | `train_student.py`, `summarize_students.py` | `results/student_eval_{tight,bfi}.tsv`, `results/student_summary_by_variant.tsv`, `checkpoints.tar` asset |
| Accuracy statistics | `bootstrap_cis.py`, `pervideo_f2.py`, `stratified_results.py` | `results/student_bootstrap_cis.tsv`, `results/pervideo_f2_test.tsv`, `results/stratified_test.tsv` |
| Smoothing, gate, alarm level | `smooth_student_outputs.py`, `event_metrics.py`, `alarm_op_rule_val.py`, `cascade_simulation.py`, `gate_candidates_val.py`, `classify_only_alarm_val.py`, `deployed_cascade_metrics.py` | `results/event_metrics.tsv`, `results/alarm_*.tsv`, `results/cascade_sim.tsv`, `results/gate_*.tsv`, `results/classify_only_alarm_*.tsv`, `results/deployed_cascade_test_centercrop.tsv`, `results/deployment_gates_centercrop.tsv` |
| Export and quantisation | `export_students.py`, `export_fidelity_val.py`, `quantize_qdq_int8.py`, `export_tflite_imx.py` | `models.tar` asset, `results/export_manifest*.tsv`, `results/export_fidelity_val.tsv` |
| Board measurement | `board_bench.py` (any board, onnxruntime: latency, full-test dumps), `jetson_measure.py` (TensorRT + tegrastats), `capture_pipeline_power.py` (reference example of a camera window with whole-unit power from the Shelly plug), `score_device_dumps.py` | `results/device/`, `results/device_latency.tsv`, `results/device_eval_*.tsv` |
| Tables and figures | `make_camera_policy_table.py`, `deployment_budget.py`, `make_pareto_figure.py`, `make_sweep_figure.py`, `make_pervideo_figure.py`, `make_ablation_readable.py` | `results/camera_policy_rows.tsv` and the paper's figures |

The camera-window rows of the paper's power table were produced by per-board capture and scheduler scripts with fixed policies and power
collection contracts; those exact scripts, their hashes and the acceptance records sit next to each window in the per-window evidence tree,
which is not included here (see the repository README). `capture_pipeline_power.py`
here is the generic reference implementation, not the per-board runtime of those rows. The i.MX95 QAT2 route (NPU and CPU scripts, models)
is documented in the same evidence tree.
