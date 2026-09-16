# ECG-FIX CLOCS deployment on 204

Pinned deployment inputs:

- ECG-FIX repository: `zackeberger/ecg-fix`
- ECG-FIX commit: `991a31f14c94f72d5658bc172eae5736ba2fa149`
- Hugging Face repository: `doprakah/ecg-fix-weights`
- Hugging Face revision: `6e2e3e6fb767df47e0b31e14ca3d73e1088c5134`
- CLOCS file: `best_weights_clocs`
- CLOCS SHA256: `039975cf563e76dd25a7975abfe1b74ff37308bd1e9fb24aaf6282f4ffdc5805`

Server paths are under `/root/107552503710-1`:

- source: `external_models/ecg-fix`
- checkpoint: `model_weights/ecg-fix/best_weights_clocs`
- isolated environment: `envs/ecg-fix`
- deployment metadata: `deployment_logs/ecgfix-clocs-deployment.txt`

Run the strict loader and forward-pass audit with:

```bash
/root/107552503710-1/envs/ecg-fix/bin/python \
  /root/107552503710-1/src/CLEAR-HUG/experiments/q1_ecgfix_clocs_deploy_204_20260916/verify_deployment.py
```

This deployment does not start evaluation while the three GPUs are occupied by
the MERL campaign. ECG-FIX preprocessing/evaluation must use the benchmark's
own split and linear-probe pipeline rather than the legacy CLOCS runner.
