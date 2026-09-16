# MERL remaining five-task campaign

This campaign runs the released MERL ResNet-18 encoder on PTB-XL
Superdiagnostic, Subdiagnostic, Form and Rhythm plus CSN, using official MERL
splits, seeds 42/46/55 and 1%/10%/100% labels. CPSC2018 was completed in the
preceding campaign and is not repeated.

The official MERL CSV files are the sole source of record membership, order and
labels. Existing CLEAR-HUG NumPy arrays are deliberately not used because their
PTB-XL label-column order differs from the released MERL CSV order.

Each of three V100 GPUs receives five size-balanced task-seed groups (15 units).
The formal run is hosted on `10.109.118.204:10092` under
`/root/107552503710-1`; missing immutable raw waveforms and the released encoder
are relayed read-only from the 172 instance and verified by archive SHA256 plus
extracted file counts.
The model and optimizer follow the released MERL downstream configuration:
batch size 16, Adam `1e-3`, weight decay `1e-4`, 100 epochs, and the released
per-batch plus per-epoch scheduler stepping. The primary result is selected by
validation Macro AUROC; the released-code max-test-over-epochs metric is stored
separately and must not be used as the formal comparison.

- MERL source commit: `2a38649285e16eff75b69aeb64f2366b380c1a9e`
- Encoder SHA256: `38ba669c2cc319670c4172d8c292f123e86b8e7106b1a68bb7e10dd89f09daf5`
