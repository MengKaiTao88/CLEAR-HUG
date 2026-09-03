# Six-task, ten-seed CLEAR-HUG / DeepSets / HiLAR campaign

This directory freezes the protocol for `q1-clear-deepsets-hilar-10seed-20260903`.
Seeds are 42 through 51. All three models select checkpoints using validation
Macro AUROC. Formal-test scripts refuse to run without a campaign-wide gate
covering all 60 task-seed units. Archived prediction scores are not accepted.

`protocol.py` is the source of truth for task metadata and the balanced
three-node assignment. `run_train_val_task_seed.sh` creates a single frozen
unit, while `run_specs_queue.sh` serializes units on one GPU. After collecting
all train/validation manifests, `pretest_gate.py` authorizes formal inference.
Final metrics and patient-cluster bootstrap statistics are recomputed by
`analyze_results.py` from the saved probabilities.
