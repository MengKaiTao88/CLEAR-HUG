# Mehari--Strodthoff SimCLR/BYOL deployment on 204

Pinned source:

- repository: `https://github.com/tmehari/ecg-selfsupervised`
- commit: `956c6c17c9496b2ba459638c613c6efc148b95df`
- architecture: `xresnet1d50`
- public checkpoints: `xresnet1d50_simclr_rrc_to_on_all.pt` and
  `xresnet1d50_byol_rrc_to_on_all.pt`
- upstream pretraining data: CINC, Zheng, and Ribeiro ECG collections
- upstream augmentations: RandomResizedCrop and TimeOut

Remote locations, all below `/root/107552503710-1`:

- source: `external_models/ecg-selfsupervised`
- checkpoints: `model_weights/ecg-selfsupervised`
- audit: `deployment_logs/ecg-selfsupervised-simclr-byol-audit.json`

These checkpoints implement the same SimCLR and BYOL algorithms named in the
MERL paper, but they must not be described as the exact MERL-table baselines.
MERL cites the original algorithm papers, uses its own uniform benchmark and
pretraining claims, and does not identify these Mehari--Strodthoff checkpoints.

The loader audit can use the already pinned Torch runtime on 204:

```bash
/root/107552503710-1/envs/ecg-fix/bin/python \
  /root/107552503710-1/src/CLEAR-HUG/experiments/q1_ecgselfsupervised_deploy_204_20260916/verify_simclr_byol.py \
  --root /root/107552503710-1
```
