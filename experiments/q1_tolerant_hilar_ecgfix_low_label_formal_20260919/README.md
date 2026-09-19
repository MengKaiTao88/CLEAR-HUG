# TolerantECG + HiLAR 1%/10% formal campaign

Runs the locked HILA-K and LRA architecture for fractions 0.01 and 0.1,
seeds 42/46/55, and all six tasks. Each run uses the exact low-label subset
indices saved by the matching TolerantECG baseline checkpoint. Hyperparameters
and validation Macro-AUROC selection remain frozen from the 100% campaign.
After validation checkpoints are complete, they are evaluated once on the
already cached formal-test features without test-time tuning.
