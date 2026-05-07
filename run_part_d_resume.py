from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from asr_project.commonvoice import load_commonvoice_test_rows
from asr_project.config import ProjectConfig
from asr_project.evaluate import evaluate_transcriptions_tsv
from asr_project.normalize_he import reference_normalize, transcription_normalize
from asr_project.transcribe import WhisperHebrewTranscriber, transcribe_benchmark_to_tsv


def main():
    project_root = Path(__file__).resolve().parent
    cfg = ProjectConfig.default(project_root)
    cfg.ensure_dirs()

    noisy_wav_count = len(list(cfg.noisy_dir.glob("*.wav")))
    if noisy_wav_count == 0:
        raise RuntimeError(f"No noisy wav files found under {cfg.noisy_dir}")

    rows = load_commonvoice_test_rows(cfg.commonvoice_test_tsv)
    noisy_rows = [
        SimpleNamespace(path=f"{r.filename_stem}.wav", sentence=r.sentence, filename_stem=r.filename_stem)
        for r in rows
    ]

    transcriber = WhisperHebrewTranscriber(
        model_name_or_path=cfg.model_name_or_path,
        device=cfg.device,
        compute_type=cfg.compute_type,
        language=cfg.language,
    )

    part_d_transcriptions_tsv = cfg.outputs_dir / "part_d_noisy_transcriptions.tsv"
    transcribe_benchmark_to_tsv(
        rows=noisy_rows,
        clips_dir=cfg.noisy_dir,
        output_tsv=part_d_transcriptions_tsv,
        transcriber=transcriber,
        resume=True,
    )

    # Guard against duplicate filenames if prior interrupted runs appended duplicates.
    df_noisy = pd.read_csv(part_d_transcriptions_tsv, sep="\t", encoding="utf-8", dtype=str).fillna("")
    dedup_df = df_noisy.drop_duplicates(subset=["Filename"], keep="first")
    if len(dedup_df) != len(df_noisy):
        dedup_df.to_csv(part_d_transcriptions_tsv, sep="\t", index=False, encoding="utf-8")

    part_d_metrics_tsv = cfg.outputs_dir / "part_d_metrics.tsv"
    part_d_alignment_log_tsv = cfg.outputs_dir / "part_d_alignment_log.tsv"
    part_d_errors_tsv = cfg.outputs_dir / "part_d_frequent_errors.tsv"

    metrics_df_d, _ = evaluate_transcriptions_tsv(
        transcriptions_tsv=part_d_transcriptions_tsv,
        output_metrics_tsv=part_d_metrics_tsv,
        output_alignment_log_tsv=part_d_alignment_log_tsv,
        output_frequent_errors_tsv=part_d_errors_tsv,
        reference_normalize=reference_normalize,
        transcription_normalize=transcription_normalize,
        frequent_errors_min_count=2,
    )

    total_row = metrics_df_d[metrics_df_d["Filename"] == "TOTAL"]
    print(total_row.to_string(index=False))
    print(f"Noisy wav files: {noisy_wav_count}")
    print(f"Transcriptions: {part_d_transcriptions_tsv}")
    print(f"Metrics: {part_d_metrics_tsv}")
    print(f"Frequent errors: {part_d_errors_tsv}")


if __name__ == "__main__":
    main()
