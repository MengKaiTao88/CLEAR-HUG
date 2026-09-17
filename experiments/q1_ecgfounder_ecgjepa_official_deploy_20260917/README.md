# Official ECGFounder and ECG-JEPA deployment on server 204

Pinned official sources and checkpoints:

- ECGFounder source commit `04edac702b61c91face519774ddcc0cd712fef23`.
  The official 12-lead checkpoint comes from Hugging Face revision
  `d9b1793951b2342f5f7e84f1ac03cd37f8a08724` and has SHA256
  `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`.
- ECG-JEPA source commit `d937ad2c2c8a1e22856ce7e4a23a30f84a71217c`.
  The official multi-block epoch-100 checkpoint is Google Drive file
  `1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l` and has SHA256
  `61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e`.

Remote locations are below `/root/107552503710-1`:

- sources: `external_models/ECGFounder`, `external_models/ECG_JEPA`;
- checkpoints: `model_weights/ECGFounder/12_lead_ECGFounder.pth` and
  `model_weights/ECG_JEPA/multiblock_epoch100.pth`;
- audit: `results/q1-ecgfounder-ecgjepa-official-deploy-20260917/deployment-manifest.json`.

`verify_deployment.py` checks source markers and checkpoint hashes, invokes the
official loaders, freezes the encoders, and performs dummy forward passes using
the official input shapes. ECGFounder expects 12 leads at 500 Hz for 10 seconds;
the released ECG-JEPA multi-block loader expects the official reduced 8-lead,
250 Hz, 10-second input.
