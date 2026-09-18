# HiLAR six-task, three-fraction rerun (2026-09-18)

Fresh end-to-end HiLAR method rerun on six tasks, fractions 1%/10%/100%, and seeds 42/46/55 (54 units).
Each unit trains a fresh HILA/DeepSets adapter for 100 epochs, selects by validation Macro AUROC, caches exactly paired frozen train/validation features, and trains a fresh no-anchor HiLAR residual classifier. A background supervisor creates the global pre-test gate only after all 54 train/validation units pass, evaluates every formal test unit once, and writes `summary.json`.

This is the native HiLAR method-level protocol, not a claim that HiLAR uses the same single-layer frozen probe as representation-only baselines.

Server campaign path:
`/root/107552503710-1/results/q1-hilar-rerun-six-task-three-fraction-seeds42-46-55-20260918`
