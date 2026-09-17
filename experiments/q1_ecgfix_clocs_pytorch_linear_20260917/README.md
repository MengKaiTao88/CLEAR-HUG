# ECG-FIX CLOCS PyTorch GPU linear probe

This campaign reuses the audited frozen CLOCS embeddings from
`q1-ecgfix-clocs-six-task-three-fraction-seeds42-46-55-20260916` and trains only
a randomly initialized PyTorch `Linear` head. It is deliberately separate from
the released ECG-FIX sklearn/LBFGS results.

Protocol: six tasks, fractions 1%/10%/100%, seeds 42/46/55, one seed per GPU.
The train subset is selected with the same seeded `torch.randperm` convention as
the released evaluator. Both CLOCS views are training examples; validation and
test probabilities average the two views. Embeddings are standardized using the
selected training subset. The optimizer is AdamW (lr 1e-3, weight decay 1e-4),
batch size 256, at most 100 epochs, and early stopping patience 12. Validation
macro AUROC is the sole checkpoint-selection metric.

Run `deploy_start.py` locally. It deploys safely but will leave the campaign in
`waiting_for_gpu` until exactly three CUDA devices are visible.
