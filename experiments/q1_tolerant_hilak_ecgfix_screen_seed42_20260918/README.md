# TolerantECG HILA-K on the first ECG-FIX protocol

This validation-only screen uses the exact first TolerantECG campaign rather
than the later MERL reordering:

- original ECG-FIX train/validation splits and preprocessing;
- the frozen seed-42, 100%-label TolerantECG linear head;
- the original per-dimension train mean/std from that head's checkpoint;
- batch size 256, AdamW at 1e-3, fixed learning rate, at most 100 epochs;
- validation Macro AUROC selection and no test access.

The same baseline mean/std standardizes both the full-view 768D representation
and each keep-one-lead 768D representation. The encoder and baseline head remain
frozen. The residual head is zero-initialized and the class-wise gate starts at
0.1, preserving exact baseline logits at initialization while allowing the
residual head to receive a first-step gradient.

The frozen architecture is additionally extended, without modification, to
`PTBXL_sub` and `PTBXL_rhythm` at seed 42 and 100% labels. These two tasks use
the same validation-only protocol and are launched by `run_sub_rhythm.sh`.
