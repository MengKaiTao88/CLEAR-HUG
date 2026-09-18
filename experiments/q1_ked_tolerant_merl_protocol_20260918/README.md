# KED / TolerantECG on the exact MERL six-task protocol

Campaign: `q1-ked-tolerant-merl-six-task-three-fraction-seeds42-46-55-20260918`.

Models: frozen KED and TolerantECG encoders. Tasks use the exact official MERL
train/validation/test CSV files for PTB-XL superdiagnostic, subdiagnostic, form,
rhythm, CPSC2018, and CSN. Label fractions are 1%, 10%, and 100%, with seeds
42, 46, and 55. Low-label rows are selected exactly with MERL's
`train_test_split(..., train_size=ratio/100, random_state=seed)` convention.

Only `Linear(768, classes)` is trained. AdamW uses lr `1e-3`, weight decay
`1e-4`, and batch size 16. Training runs for at most 100 epochs with 5 warmup
epochs, cosine annealing, and patience 12. Validation Macro AUROC selects the
checkpoint. Test is evaluated exactly once after selection.

The preparation stage reorders the already-audited frozen embeddings by ECG
identity; it does not rerun or alter either encoder. Every split and subset has
record-ID, label, source-index, and official-CSV hashes.
