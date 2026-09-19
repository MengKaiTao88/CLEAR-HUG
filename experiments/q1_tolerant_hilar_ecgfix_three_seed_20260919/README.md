# Locked three-seed TolerantECG HiLAR validation replication

This campaign reuses the completed seed-42 checkpoints and trains only seeds
46 and 55 for all six tasks. It preserves the locked first ECG-FIX protocol:
100% labels, AdamW (`1e-3`, weight decay `1e-4`), batch size 256, fixed LR,
maximum 100 epochs, patience 12, and validation Macro AUROC selection. Test
data are never loaded.

For each task and new seed, `train.py` first trains HILA-K from the matching
frozen TolerantECG baseline head and then trains LRA from that selected HILA-K
checkpoint. Existing lead-masked and heartbeat-local encoder features are
reused only after record-index and label hashes are verified.
