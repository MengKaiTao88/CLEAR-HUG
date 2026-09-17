# Custom ST-MEM seed-123 comparison

This campaign deliberately evaluates the user-supplied
`pretrained_checkpoint.pt` (SHA256 `ad7e8527...ac09df9`). It is **not** the
official public ST-MEM ViT-B encoder. Its embedded metadata identifies a
ST-MEM ViT-S/50 model, seed 123, trained for 40 epochs by the historical
unified baseline implementation on the PTB-XL five-class training split.

The encoder expects raw 12-lead, 100 Hz, 10-second arrays (`12 x 1000`) and
produces 384-dimensional representations. The six downstream tasks use the
same audited record-level splits, label order, fractions (1%, 10%, 100%), and
seeds (42, 46, 55) as the CLOCS/ECGFounder/ECG-JEPA comparisons. Only a random
PyTorch linear head is trained with AdamW lr 1e-3, weight decay 1e-4, batch 256,
up to 100 epochs, patience 12, and validation Macro AUROC checkpoint selection.
