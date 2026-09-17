# ECGFounder and ECG-JEPA frozen-encoder comparison

Campaign: `q1-ecgfounder-ecgjepa-six-task-three-fraction-seeds42-46-55-20260917`.

This campaign evaluates the pinned official pretrained encoders on PTB-XL Form,
Superdiagnostic, Subdiagnostic and Rhythm, CPSC2018, and CSN at 1%, 10%, and
100% labels with seeds 42, 46, and 55. It reuses the audited record-level splits
and label order from the completed ECG-FIX/CLOCS campaign.

Model-specific preprocessing remains official: ECGFounder receives globally
z-normalized 12-lead 500 Hz signals; ECG-JEPA receives I, II, V1-V6 resampled
with SciPy Fourier resampling to 250 Hz. Encoders are frozen and cached once.
The downstream protocol is the same PyTorch linear head used for CLOCS:
AdamW (lr 1e-3, weight decay 1e-4), batch size 256, at most 100 epochs,
patience 12, and validation Macro AUROC model selection. Formal test is run once
from the validation-selected checkpoint.
