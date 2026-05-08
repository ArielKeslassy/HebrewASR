# Hebrew Automatic Speech Recognition (ASR) Pipeline
🏆 **Final Grade: 97/100**

## Overview
This repository contains a full Hebrew Automatic Speech Recognition (ASR) pipeline built for a final academic project. It covers data loading, raw transcription using `faster-whisper`, custom text normalization, and noise robustness evaluation on the CommonVoice Hebrew dataset.

## Problem Statement
Automatic Speech Recognition (ASR) for the Hebrew language presents unique challenges compared to English. Hebrew is a morphologically rich language where vowels are often omitted in written text (unvocalized), and digit-to-word conversions heavily depend on grammatical gender. Off-the-shelf models often output raw numbers, punctuation, or diacritics (niqqud) that mismatch the ground truth transcriptions. This leads to artificially inflated Word Error Rates (WER) even when the model's acoustic phonetic predictions are perfectly accurate. 

This project aims to evaluate an out-of-the-box ASR model on Hebrew data, identify its primary error modes, design an iterative text normalization pipeline to bridge the gap between spoken and written Hebrew, and finally, stress-test the model's acoustic robustness against synthetic background noise.

## Theoretical Background
This project bridges several core concepts in speech processing and natural language processing:

1. **Word Error Rate (WER) & Sequence Alignment**: 
   To accurately evaluate the ASR model, we use Dynamic Programming (similar to Levenshtein distance) to align the predicted transcript with the reference text. This algorithm calculates the optimal number of Substitutions ($S$), Deletions ($D$), and Insertions ($I$). The WER is defined as $WER = (S + D + I) / N_{gt}$ (where $N_{gt}$ is the number of words in the ground truth). We also compute Recall, Precision, and F1-scores based on these alignment counts.
2. **Text Normalization**: 
   Due to orthographic variations (e.g., differences in spelling equivalence, presence of diacritics), raw ASR outputs often penalize the model unfairly. We implemented a 5-stage NLP normalization pipeline that standardizes punctuation, strips diacritics, maps lexical equivalences, and converts numerical digits to their correct Hebrew word representations using `num2words`.
3. **Signal-to-Noise Ratio (SNR) & Acoustic Robustness**:
   Real-world ASR systems must handle environmental noise. We simulate this by downsampling our audio to 16kHz and injecting background interference from the MUSAN dataset. We dynamically scale the background noise by calculating the signal energy and applying a scaling factor $\alpha$ to achieve a target Signal-to-Noise Ratio (SNR) in the $6-12 \text{ dB}$ range.

## Results & Insights
Our systematic evaluation yielded the following progression on the CommonVoice test set:

| Stage | Description | WER | F1-Score |
| :--- | :--- | :--- | :--- |
| **Stage B** | Raw Baseline (No Normalization) | 35.36% | 66.20% |
| **Stage C.1** | Diacritics Stripped | 23.27% | - |
| **Stage C.2** | Punctuation Standardized | 11.42% | - |
| **Stage C.4** | Full Lexical Normalization | 8.05% | 92.58% |
| **Stage C.5** | Digit-to-Word Conversion | **6.94%** | **93.50%** |
| **Stage D** | Noisy Audio (SNR 6-12dB) | 12.70% | 88.33% |

**Key Takeaways:**
* **Text Normalization is Crucial:** Over 80% of the baseline errors were "artificial" orthographic mismatches. By standardizing the text, we reduced the WER from 35.36% to a highly accurate 6.94%.
* **Grammatical Gender Limitations:** Even with `num2words`, Hebrew's complex grammatical gender rules for numbers remain a persistent source of error.
* **Acoustic Robustness:** The introduction of moderate background noise ($6-12 \text{ dB}$ SNR) degraded performance back to 12.70% WER, demonstrating that while the text pipeline is strong, the acoustic model still struggles to isolate phonemes in noisy environments.

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

Install the required dependencies using:
```bash
pip install -r requirements.txt
```


## Data Paths
> [!NOTE]
> The `data/` directory is excluded from this repository (via `.gitignore`) because audio files and large datasets are too heavy for GitHub. If you wish to run the code locally, please download the CommonVoice Hebrew and MUSAN datasets and place them in the folder structure described below. The `data/outputs/` directory will be generated automatically when you run the scripts.

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
python3 run_part_d.py --musan_root "/path/to/musan" --modulo_val 2 --seed 42
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


## Notebook Comparison Graphs
`ASR_Final_Project.ipynb` includes Part D analysis cells (after stage D outputs) that:
- load Part B/Part C/Part D totals,
- compare clean vs noisy metrics,
- plot WER/F1 comparisons and stage progression.

## Reproducibility Tips
- Keep `resume=True` for long transcription stages.
- If interrupted in stage D, rerun `run_part_d_resume.py`.
- Verify MUSAN `.wav` availability before stage D.
