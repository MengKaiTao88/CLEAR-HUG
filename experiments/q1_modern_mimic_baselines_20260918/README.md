# KED, D-BETA, and TolerantECG frozen linear probes

Campaign: `q1-modern-mimic-baselines-six-task-three-fraction-seeds42-46-55-20260918`.

The campaign evaluates three MIMIC-IV-pretrained released encoders on the same
ECG-FIX record-level splits for PTB-XL form/super/sub/rhythm, CPSC2018, and CSN.
For every model, the encoder is frozen and only a randomly initialized PyTorch
`Linear(768, classes)` head is trained. Fractions are 1%, 10%, and 100%; seeds
are 42, 46, and 55. AdamW uses learning rate `1e-3`, weight decay `1e-4`, batch
size 256, at most 100 epochs, and early-stopping patience 12. Validation Macro
AUROC selects the checkpoint and formal test is evaluated once.

KED uses the 1,341,207,980-byte checkpoint from official Zenodo record 14881564
(the identical ECG-FIX filename mirror is uploaded as `ked.pt`). D-BETA uses the
public ECG-FIX mirror because the author Hugging Face repository requires manual
access acceptance; its exact SHA256 is recorded in every embedding manifest.
TolerantECG uses `ndhuynh02/TolerantECG/TolerantECG_encoder.pth` and the official
ConvNeXtV2 source at commit `f38ad823da11da4336d8a057cc5ff2c6f1693da8`.

Each model owns one GPU. Embedding extraction is resumable per dataset/split;
probe training is resumable per model/seed/task/fraction. Manifests record the
checkpoint hash, source-index hash, and label hash so cross-model pairing can be
audited before reporting results.
