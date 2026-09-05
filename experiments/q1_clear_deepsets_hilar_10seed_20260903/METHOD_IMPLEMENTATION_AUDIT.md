# HiLAR method implementation audit

This note records the implementation used by the frozen six-task campaign
`q1-clear-deepsets-hilar-10seed-20260903`. It distinguishes the formal model
from abstractions that appeared in draft method text.

## 1. HILA is the identity-aware masked DeepSets baseline

The formal HILA branch operates on the twelve contextualized Lead-CLS tokens
produced by CLEAR, not on a heartbeat-by-lead tensor. Its input is
`lead_cls` with shape `B x 12 x 768` and its validity mask is `B x 12`.

For lead `l`, a learned embedding `e_l` has 32 dimensions. The element encoder
and post-pooling transform are

```text
phi: Linear(800, 1408, bias=True)
     GELU
     Linear(1408, 1408, bias=True)
     GELU

rho: Linear(1408, 768, bias=True)
     GELU
```

The exact computation is

```text
u_l = phi(concat(lead_cls_l, e_l))
c   = rho(masked_mean_l(u_l))
z_HILA = Linear(768, C, bias=True)(c)
```

Thus `c` is 768-dimensional. Masked mean is applied after `phi` and before
`rho`. HILA has no LayerNorm, Dropout, or additional residual connection in
the aggregation adapter. All three Linear layers use bias. CLEAR itself was
constructed with transformer and embedding dropout of 0.1, but those belong
to the frozen backbone rather than HILA.

There is no separate heartbeat-sequence function `g_psi({c_j})` in the formal
baseline. The effective `g_psi` is a single `Linear(768, C)` classifier applied
to the one record-level HILA vector `c`. A draft that defines heartbeat-level
`c_j` followed by heartbeat pooling does not describe the formal model.

## 2. CLEAR-to-HiLAR interface

Each record is represented by 180 QRS tokens (`12 leads x 15 heartbeat slots`),
each with 96 waveform samples. CLEAR returns twelve Lead-CLS tokens followed by
180 contextualized QRS tokens, all 768-dimensional.

The formal HILA prediction path uses:

```text
Lead-CLS: B x 12 x 768
lead_valid[b,l] = any_j(in_time[b,l,j] > 0)
```

The feature cache restores the QRS token order and saves:

```text
local:          N x 15 x 12 x 768, float16
shared:         N x 15 x 768,      float16
valid:          N x 15 x 12,       bool
baseline_logits:N x C,             float32
labels:         N x C,              float32
```

The pair mask is `valid[b,j,l] = in_time[b,l,j] > 0`. The LRA heartbeat mask is
`beat_valid[b,j] = any_l(valid[b,j,l])`. The HILA lead mask and LRA heartbeat
mask are therefore distinct reductions of the same heartbeat-lead validity
grid. The implementation fixes `J=15`; it does not accept a dynamic-length
heartbeat sequence at this interface.

HILA masks only at whole-lead aggregation: a lead is retained when any of its
15 heartbeat slots is valid. The inspected forward path passes
`key_padding_mask=None` into CLEAR, so an invalid heartbeat slot inside an
otherwise retained lead is not independently removed by HILA after encoding.
LRA does exclude that individual heartbeat-lead position through `valid`.

The aggregation code does not require invalid cached local vectors to be zero,
because it removes them through `valid`. The inspected model code does not
establish that every waveform padding token is numerically zero, so the paper
should not make that stronger claim without a separate dataset audit.

The cache also computes a per-heartbeat `shared` vector by applying the same
trained DeepSets adapter to each heartbeat's twelve local tokens. The final
`parameter-matched-direct` LRA receives this array for interface compatibility
but does not use it; `shared` is used only by the reconstruction variants.

## 3. Training and freezing

For HILA/DeepSets, a fresh downstream model is constructed and the released
CLEAR checkpoint is loaded. The CLEAR backbone, including token, spatial, and
temporal embeddings, is parameter-frozen. Only `adapter` (lead embedding,
`phi`, and `rho`) and `mlp_head` are trainable. The downstream classifier is
freshly constructed and trained with the adapter.

The formal campaign trains this stage for 100 epochs and selects the checkpoint
by validation Macro AUROC. It uses batch size 256, learning rate `5e-3`, weight
decay `0.05`, and ten warm-up epochs. Parameter freezing does not force the
backbone into evaluation mode during HILA optimization; ordinary `model.train()`
semantics can keep backbone Dropout active even though its weights are frozen.
The paper can accurately say "parameter-frozen CLEAR backbone."

For every task and seed, the selected DeepSets checkpoint is then frozen and
used to cache local features and baseline logits. The matching LRA is trained
on those cached values. HILA and LRA are therefore sequentially trained, not
jointly optimized. The strict pairing is

```text
DeepSets seed s -> frozen checkpoint/features/logits -> LRA seed s
```

During LRA training no CLEAR, HILA adapter, or HILA classifier parameter is
updated.

## 4. Lead identities are not shared between HILA and LRA

HILA contains `nn.Embedding(12, 32)`. The direct LRA contains a second,
independent `nn.Embedding(12, 32)`. They are separately initialized and trained;
there is no parameter sharing or inheritance. The manuscript therefore must
not state that both branches use one shared learned lead embedding or a single
learned semantic coordinate system.

PyTorch's default `nn.Embedding` initialization is used (independent standard
normal weights). Both embedding tables are trainable in their respective
stages. Missing leads do not contribute an embedding or a zero-token element to
the masked aggregation; the corresponding element is excluded by the mask.

## 5. Final local residual adapter

The selected `parameter-matched-direct`, no-anchor LRA is:

```text
Direct token transform:
  concat(local[768], independent lead embedding[32]) -> 800
  Linear(800, 512, bias=True)
  GELU
  LayerNorm(512)
  Linear(512, 768, bias=True)

Residual encoder:
  LayerNorm(768)
  Linear(768, 256, bias=False)
  GELU
  Dropout(0.1)
  Linear(256, 128, bias=False)

Pooling and prediction:
  q_j = masked_mean over leads of the 128-dimensional tokens
  q   = masked_mean over valid heartbeats of q_j
  delta = Linear(128, C, bias=True)(q)
  z_HiLAR = frozen_baseline_logits + delta
```

Only the final `Linear(128, C)` weight and bias are zero-initialized. The direct
token transform and residual encoder use their normal random/default
initializations. The existing unit test verifies exact equality of HiLAR and
baseline logits at initialization and verifies that a classification gradient
opens the zero-initialized branch. The HILA branch remains frozen throughout
this optimization.

The two-stage pooling gives each valid heartbeat one pooled vector before
record-level averaging. It is equal to a flat token mean only when each valid
heartbeat has the same number of valid leads. With variable lead availability,
it gives heartbeats equal weight instead of weighting heartbeats by their count
of observed leads. This implementation property supports a cautious design
explanation, but no dedicated flat-pooling ablation was found.

## 6. Objective and prediction

Both stages use multi-label binary cross entropy on logits. LRA calls
`binary_cross_entropy_with_logits`, which is equivalent to
`BCEWithLogitsLoss`. The formal implementation supplies no class weights,
`pos_weight`, label smoothing, or threshold to the training loss. Probabilities
are obtained with one sigmoid for AUROC/AUPRC. Thresholding is not used to
compute those two ranking metrics.

The final campaign sets `anchor_lambda=0`; consequently its LRA objective is
only classification BCE. The optional squared-logit correction penalty is not
active.

## 7. Permutation property and identity claim

HILA is invariant to a joint permutation of the triples
`(Lead-CLS value, validity, lead ID)`. A unit test permutes all three together
and checks that the output is unchanged. Permuting feature values without their
lead IDs is a different intervention and is not required to preserve the
output.

The architectural difference from identity-free DeepSets is exactly

```text
phi(concat(h_l, e_l))  versus  phi(h_l).
```

However, the formal six-task campaign contains no identity-free HILA baseline.
The implemented `parameter-matched-direct-no-lead` variant removes identity
from the LRA path, not from HILA, and it was not part of the final ten-seed
matrix. The paper can call the formal aggregator identity-aware based on its
definition, but a causal performance claim about HILA's lead embedding still
requires a matched HILA-without-identity experiment.

## 8. Exact trainable parameter counts

HILA adapter components:

| Component | Parameters |
|---|---:|
| Lead embedding `12 x 32` | 384 |
| `Linear(800,1408)` | 1,127,808 |
| `Linear(1408,1408)` | 1,983,872 |
| `Linear(1408,768)` | 1,082,112 |
| HILA adapter total | 4,194,176 |
| HILA classifier | `769C` |

LRA components:

| Component | Parameters |
|---|---:|
| Independent lead embedding | 384 |
| Direct MLP, including LayerNorm | 805,120 |
| Residual encoder | 230,912 |
| Zero-initialized residual head | `129C` |
| LRA total | `1,036,416 + 129C` |

The trainable parameters above the frozen CLEAR backbone are therefore
`5,230,592 + 898C`. Per formal task:

| Task | C | HILA adapter + classifier | LRA | Combined above CLEAR |
|---|---:|---:|---:|---:|
| Superdiagnostic | 5 | 4,198,021 | 1,037,061 | 5,235,082 |
| Subdiagnostic | 23 | 4,211,863 | 1,039,383 | 5,251,246 |
| Form | 19 | 4,208,787 | 1,038,867 | 5,247,654 |
| Rhythm | 12 | 4,203,404 | 1,037,964 | 5,241,368 |
| CPSC2018 | 9 | 4,201,097 | 1,037,577 | 5,238,674 |
| CSN | 38 | 4,223,398 | 1,041,318 | 5,264,716 |

These are exact counts from the declared layer shapes. The end-to-end model
count is `frozen CLEAR parameter count + combined above CLEAR`. The current
local campaign manifests do not store the frozen CLEAR parameter count, so an
absolute end-to-end total should be added only after a model inventory audit.

## 9. Manuscript consequences

The current formal implementation supports the following method structure:

1. CLEAR produces twelve contextualized Lead-CLS summaries plus a restored
   heartbeat-by-lead local token grid.
2. HILA applies identity-aware, permutation-invariant DeepSets aggregation to
   the twelve Lead-CLS summaries and produces record logits.
3. LRA independently encodes the local heartbeat-by-lead grid, pools leads and
   then heartbeats, and adds a zero-start correction to the frozen HILA logits.

Accordingly, phrases such as "heartbeat-wise HILA", `g_psi({c_j})`, "shared
HILA/LRA lead embedding", and "joint end-to-end training" do not describe the
formal six-task model. Heartbeat-wise organization belongs to LRA in this
implementation.
