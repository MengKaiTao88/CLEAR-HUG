# TolerantECG HILA-K validation screen

This experiment keeps the official TolerantECG encoder and the existing
MERL-protocol linear probe frozen. It caches twelve keep-one-lead views after
the established ECG-FIX preprocessing and trains the audited HILA dimensions:
`Embedding(12,32)`, `800 -> 1408 -> 1408`, mean over leads, and
`1408 -> 768`.

The residual classifier is zero-initialized and the class-wise gate starts at
`0.1`. Therefore initial logits exactly equal the baseline while the residual
head receives a gradient on the first update. Tests cover both properties.

The screen is deliberately limited to seed 42, 100% labels, and
`superdiagnostic`, `form`, `cpsc2018`, and `csn`. It uses train and validation
only. No test arrays are loaded or evaluated during model selection.

