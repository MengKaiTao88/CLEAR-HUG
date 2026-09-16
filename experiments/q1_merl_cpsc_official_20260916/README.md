# Official MERL CPSC2018 reproduction

This campaign runs the released MERL ResNet-18 encoder on the official CPSC2018
split (`4950/551/1376`) at 1%, 10% and 100% labels with the released linear-probe
configuration: batch size 16, Adam at `1e-3`, weight decay `1e-4`, 100 epochs,
and seed behavior from the public source.

Provenance:

- Source: `https://github.com/cheliu-computation/MERL-ICML2024`
- Source commit: `2a38649285e16eff75b69aeb64f2366b380c1a9e`
- Released encoder: `res18_best_encoder.pth`
- Encoder SHA256: `38ba669c2cc319670c4172d8c292f123e86b8e7106b1a68bb7e10dd89f09daf5`

The public MERL downstream source evaluates test data after every epoch and
reports the maximum test metric. This runner records that value as
`released_code_max_test_over_epochs`, but the primary fair-comparison output is
`validation_selected_test`, obtained from the checkpoint selected only by
validation Macro AUROC.

The released `res18_best_encoder.pth` unexpectedly contains `linear.weight`
and `linear.bias` for a 10-class pretraining head. PyTorch does not ignore a
shape mismatch under `strict=False`, so the unmodified public CPSC code fails
when constructing its 9-class head. The runner removes only these two released
head tensors and records their shapes in the model audit; all encoder tensors
remain unchanged.

The supplemental campaign `q1-merl-cpsc-official-seeds46-55-20260916` adds
seeds 46 and 55 without overwriting seed 42. For this multi-seed extension,
PyTorch, Python, NumPy and the low-label subset sampler all use the task seed.
