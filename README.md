# Hebrew Automatic Speech Recognition (ASR) Pipeline
🏆 **Final Grade: 97/100**

## Overview
This repository contains a full Hebrew Automatic Speech Recognition (ASR) pipeline built for a final academic project. It covers data loading, raw transcription using `faster-whisper`, custom text normalization, and noise robustness evaluation on the CommonVoice Hebrew dataset.

## Repository Layout
- `asr_project/`: core implementation (loading, transcription, alignment, metrics, normalization, noise augmentation).
- `data/commonvoice_he/`: CommonVoice Hebrew files and clips.
- `data/outputs/`: stage outputs (metrics, frequent errors, logs, reports).
- `data/noisy_benchmark/`: generated noisy wav files for stage D.
- `ASR_Final_Project.ipynb`: main notebook.
- `run_stage_c_iterative.py`: CLI for stage C iterative evaluation.
- `run_part_d.py`: full stage D CLI (noise generation + transcription + evaluation).
- `run_part_d_resume.py`: stage D resume CLI (transcription/evaluation only; no re-noising).

## Environment
Recommended Python: `3.10+`

Key dependencies:
- `pandas`
- `numpy`
- `scipy`
- `soundfile`
- `tqdm`
- `faster-whisper`
- `num2words`


## Data Paths
- CommonVoice test TSV: `data/commonvoice_he/test.tsv`
- CommonVoice clips: `data/commonvoice_he/clips`
Expected MUSAN structure includes `noise/`, `music/`, `speech/` and `.wav` files.

## How To Run
### Stage A - Raw transcription
Preferred: run notebook section `(A)` in `ASR_Final_Project.ipynb`.

Output:
- `data/outputs/part_a_transcriptions.tsv`

### Stage B - Baseline evaluation (no normalization)
Preferred: run notebook section `(B)`.

Outputs:
- `data/outputs/part_b_metrics.tsv`

Additional output for insights:
- `data/outputs/part_b_frequent_errors.tsv`
- `data/outputs/part_b_alignment_log.tsv`

### Stage C - Iterative normalization
Run:
```powershell
python3 run_stage_c_iterative.py
```

Outputs:
- `data/outputs/part_c_iterative_summary.tsv`

Additional outputs for insights:
- `data/outputs/part_c_iterative_report.txt`
- `data/outputs/part_c_metrics_stage*.tsv`
- `data/outputs/part_c_top10_errors_stage*.tsv`
- `data/outputs/part_c_alignment_log_stage*.tsv`

Normalization notes:
- Stage 5 adds `num2words` conversion; most mistakes after the normalization seem to come from the gender ambiguity of num2word.

### Stage D - Noise robustness
Full run:
```powershell
python3 run_part_d.py --musan_root "C:\Users\kaiia\OneDrive\Desktop\musan\musan" --modulo_val 2 --seed 42
```

Resume only (skip re-noising):
```powershell
python3 run_part_d_resume.py
```

Outputs:
- `data/outputs/part_d_noise_log.tsv`

Additional outputs for insights:
- `data/outputs/part_d_noisy_transcriptions.tsv`
- `data/outputs/part_d_metrics.tsv`
- `data/outputs/part_d_frequent_errors.tsv`
- `data/outputs/part_d_alignment_log.tsv`


## Results Summary
### Stage A
- Produced transcription file with `910` utterances (`911` lines including header).

### Stage B (Raw, no normalization)
- `WER = 0.3536`
- `Recall = 0.6633`
- `Precision = 0.6608`
- `F1 = 0.6620`

### Stage C (Iterative normalization)
- `C_stage1_diacritics_only`: `WER = 0.2327`
- `C_stage2_plus_punctuation`: `WER = 0.1142`
- `C_stage3_plus_hyphen`: `WER = 0.0844`
- `C_stage4_complete_normalization`: `WER = 0.0805`
- `C_stage5_plus_num2words`: `WER = 0.0694`

Observation:
- Most gains come from text normalization (diacritics/punctuation/hyphen).
- `num2words` further improves WER but Hebrew grammatical gender remains a known limitation.

### Stage D (Noisy audio, modulo profile 2)
- `WER = 0.1270`
- `Recall = 0.8800`
- `Precision = 0.8866`
- `F1 = 0.8833`

Observation:
- Noise degrades performance relative to best clean normalization.
- Numeric forms and lexical confusions remain dominant error classes in noisy conditions.

## Notebook Comparison Graphs
`ASR_Final_Project.ipynb` includes Part D analysis cells (after stage D outputs) that:
- load Part B/Part C/Part D totals,
- compare clean vs noisy metrics,
- plot WER/F1 comparisons and stage progression.

## Reproducibility Tips
- Keep `resume=True` for long transcription stages.
- If interrupted in stage D, rerun `run_part_d_resume.py`.
- Verify MUSAN `.wav` availability before stage D.
