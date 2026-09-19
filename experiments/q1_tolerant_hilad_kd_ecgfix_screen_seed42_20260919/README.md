# TolerantECG HILA-D / HILA-KD small validation screen

This experiment compares the already frozen HILA-K configuration with two
predefined lead representations on `PTBXL_form`, `CPSC`, and `PTBXL_rhythm` at
seed 42 and 100% labels. It uses the original ECG-FIX train/validation split,
never loads test data, and exactly retains HILA-K's AdamW/fixed-LR/batch-size/
early-stopping protocol.

- **HILA-D:** for lead `l`, use `g(x) - g(x without lead l)`.
- **HILA-KD:** concatenate the standardized keep-one-lead K representation and
  contextual D representation, then learn a `1536 -> 768` fusion projection.

K is standardized as `(k - baseline_train_mean) / baseline_train_std`. D is
standardized as `d / baseline_train_std`, which is exactly the difference of
the two standardized full and leave-one-out embeddings. Both variants retain
the frozen encoder, frozen baseline linear head, class-wise gate initialized to
0.1, and zero-initialized residual classifier, so initial logits equal the
baseline exactly.

The predeclared selection rule is to replace HILA-K only if D or KD improves
the three-task mean validation AUROC by at least 0.2 percentage points, wins on
at least two of three tasks, and has no clear task-level regression.
