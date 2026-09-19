# TolerantECG HILA-K + LRA validation screen

This is the predefined three-task (`PTBXL_form`, `CPSC`, `PTBXL_rhythm`), seed-42,
100%-label, validation-only screen for the first complete TolerantECG-compatible
HiLAR model. Test data are never loaded.

The selected HILA-K checkpoint is loaded and completely frozen. From the full
12-lead TolerantECG input, the frozen ConvNeXt temporal map before global average
pooling is retained. A deterministic lead-II derivative-energy detector locates
R peaks; each peak is mapped to the 156-step temporal map and pooled over a fixed
radius of four map positions. At most 24 heartbeat tokens are retained.

LRA applies `768 -> 512 -> 256 -> 128` to each standardized heartbeat token,
masked-means the tokens, and produces residual logits through a zero-initialized
head and a class-wise gate initialized to 0.1. Thus epoch 0 exactly reproduces
the locked HILA-K checkpoint. Only LRA parameters train. The optimizer, batch
size, fixed LR, 100-epoch cap, patience, and validation selection match HILA-K.

The HILA-K epoch-0 checkpoint is included as a selection candidate, so LRA can
never silently replace it with a worse validation checkpoint. The screen asks
whether this fixed local branch improves HILA-K consistently without any test
access; it does not reopen HILA architecture search.

After the architecture was locked on the original three tasks, it is extended
without modification to `PTBXL_super`, `PTBXL_sub`, and `CSN` by
`run_extension.sh`. The extension retains seed 42, 100% labels, all training
hyperparameters, validation-only selection, and zero test access.
