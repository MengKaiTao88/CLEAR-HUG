# HUG -> DeepSets -> HiLAR representation-space figure

This figure uses the identical single-label PTB-XL Superdiagnostic records in
all three panels.  Because the no-anchor HiLAR adapter produces a class-logit
correction rather than a same-dimensional fused latent, all panels use the
common five-dimensional class-logit decision space.  Cluster metrics are
computed in standardized 5D space, not on the t-SNE coordinates.

Example:

```powershell
python plot_hug_deepsets_hilar.py `
  --hug <hug-test-predictions.npz> `
  --paired <deepsets-hilar-test-predictions.npz> `
  --output-dir <output-directory> `
  --seed 42
```
