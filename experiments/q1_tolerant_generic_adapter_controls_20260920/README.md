# TolerantECG generic downstream adapter controls

This locked, validation-only campaign tests whether the gain of HILA-K + LRA can
be explained by downstream parameter count alone. It compares the existing
frozen-encoder LP and HiLAR results against two controls that consume **only the
global 768-D TolerantECG embedding**:

1. `parameter-matched-mlp`: a standalone `768 -> h -> h -> C` MLP head.
2. `generic-adapter`: a residual global adapter
   `z = z_LP + alpha * W_head(GELU(W_up(GELU(W_down(x)))))`, with frozen LP.

For every task, each control's trainable parameter count is matched within 1%
to the complete HILA-K + LRA trainable budget. Neither control receives
lead-masked features or heartbeat-local tokens. Seeds, splits, normalized
features, AdamW settings, batch size, early stopping, and validation Macro
AUROC selection are identical to the locked 100%-label HiLAR campaign. Test
data are not loaded or evaluated in this screening campaign.

Run on two GPUs:

```bash
nohup bash run_queue.sh 0 0 2 > gpu0.log 2>&1 &
nohup bash run_queue.sh 1 1 2 > gpu1.log 2>&1 &
```

After all 36 control runs complete:

```bash
/root/107552503710-1/.venv/bin/python summarize.py --root /root/107552503710-1
```
