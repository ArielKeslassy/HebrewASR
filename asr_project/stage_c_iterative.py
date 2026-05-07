from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .evaluate import evaluate_transcriptions_tsv
from .normalize_he import (
    NormalizationConfig,
    STAGE_C_COMPLETE,
    STAGE_C_COMPLETE_PLUS_NUMBERS,
    make_normalizer,
)


@dataclass(frozen=True)
class StageCStep:
    stage_name: str
    description: str
    reference_config: NormalizationConfig
    transcription_config: NormalizationConfig
    notes: str = ""


def build_default_stage_c_steps() -> list[StageCStep]:
    stage1_diacritics_only = NormalizationConfig(
        remove_diacritics=True,
        hyphen_to_space=False,
        remove_punctuation=False,
        word_equivalents=None,
    )
    stage2_diacritics_plus_punct = NormalizationConfig(
        remove_diacritics=True,
        hyphen_to_space=False,
        remove_punctuation=True,
        word_equivalents=None,
    )
    stage3_plus_hyphen = NormalizationConfig(
        remove_diacritics=True,
        hyphen_to_space=True,
        remove_punctuation=True,
        word_equivalents=None,
    )

    return [
        StageCStep(
            stage_name="C_stage1_diacritics_only",
            description="Diacritics removal only",
            reference_config=stage1_diacritics_only,
            transcription_config=stage1_diacritics_only,
        ),
        StageCStep(
            stage_name="C_stage2_plus_punctuation",
            description="Diacritics + punctuation normalization",
            reference_config=stage2_diacritics_plus_punct,
            transcription_config=stage2_diacritics_plus_punct,
        ),
        StageCStep(
            stage_name="C_stage3_plus_hyphen",
            description="Diacritics + punctuation + hyphen normalization",
            reference_config=stage3_plus_hyphen,
            transcription_config=stage3_plus_hyphen,
        ),
        StageCStep(
            stage_name="C_stage4_complete_normalization",
            description="Complete normalization + spelling equivalence map",
            reference_config=STAGE_C_COMPLETE,
            transcription_config=STAGE_C_COMPLETE,
        ),
        StageCStep(
            stage_name="C_stage5_plus_num2words",
            description="Complete normalization + num2words digit-to-word conversion",
            reference_config=STAGE_C_COMPLETE_PLUS_NUMBERS,
            transcription_config=STAGE_C_COMPLETE_PLUS_NUMBERS,
            notes=(
                "Known limitation: num2words may still choose the wrong grammatical gender "
                "for some Hebrew contexts."
            ),
        ),
    ]


def _safe_token(token: str) -> str:
    return token if token else "<eps>"


def _top_errors_dataframe(freq_items: Iterable[tuple[tuple[str, str], int]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (ref_w, hyp_w), count in freq_items:
        rows.append(
            {
                "RefWord": ref_w,
                "HypWord": hyp_w,
                "Count": int(count),
                "ErrorPair": f"{_safe_token(ref_w)} -> {_safe_token(hyp_w)}",
            }
        )
    return pd.DataFrame(rows, columns=["RefWord", "HypWord", "Count", "ErrorPair"])


def run_stage_c_iterative(
    transcriptions_tsv: Path,
    outputs_dir: Path,
    steps: list[StageCStep] | None = None,
    frequent_errors_min_count: int = 2,
    top_errors_eval: int = 10,
    top_errors_report: int = 5,
    summary_tsv_path: Path | None = None,
    report_txt_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if steps is None:
        steps = build_default_stage_c_steps()
    if not steps:
        raise ValueError("At least one Stage C step is required")
    if top_errors_eval < 1:
        raise ValueError("top_errors_eval must be >= 1")
    if top_errors_report < 1:
        raise ValueError("top_errors_report must be >= 1")

    summary_rows: list[dict[str, object]] = []
    per_step_top_errors: dict[str, pd.DataFrame] = {}
    report_lines: list[str] = []

    for idx, step in enumerate(steps, start=1):
        metrics_tsv = outputs_dir / f"part_c_metrics_stage{idx}.tsv"
        alignment_log_tsv = outputs_dir / f"part_c_alignment_log_stage{idx}.tsv"
        frequent_errors_tsv = outputs_dir / f"part_c_frequent_errors_stage{idx}.tsv"
        top10_errors_tsv = outputs_dir / f"part_c_top{top_errors_eval}_errors_stage{idx}.tsv"

        metrics_df, total_stats = evaluate_transcriptions_tsv(
            transcriptions_tsv=transcriptions_tsv,
            output_metrics_tsv=metrics_tsv,
            output_alignment_log_tsv=alignment_log_tsv,
            output_frequent_errors_tsv=frequent_errors_tsv,
            reference_normalize=make_normalizer(step.reference_config),
            transcription_normalize=make_normalizer(step.transcription_config),
            frequent_errors_min_count=frequent_errors_min_count,
        )

        total_row = metrics_df.loc[metrics_df["Filename"] == "TOTAL"].iloc[0]
        top_freq = total_stats.frequent_errors(
            min_count=frequent_errors_min_count,
            top_k=top_errors_eval,
        )
        top_freq_df = _top_errors_dataframe(top_freq)
        top_freq_df.to_csv(top10_errors_tsv, sep="\t", index=False, encoding="utf-8")
        per_step_top_errors[step.stage_name] = top_freq_df

        summary_rows.append(
            {
                "Stage": step.stage_name,
                "Description": step.description,
                "Notes": step.notes,
                "WER": float(total_row["WER"]),
                "Recall": float(total_row["Recall"]),
                "Precision": float(total_row["Precision"]),
                "F1-Score": float(total_row["F1-Score"]),
                "MetricsTSV": str(metrics_tsv),
                "TopErrorsTSV": str(top10_errors_tsv),
            }
        )

        report_lines.append(f"Stage {idx}: {step.stage_name}")
        report_lines.append(f"Description: {step.description}")
        if step.notes:
            report_lines.append(f"Notes: {step.notes}")
        report_lines.append(f"WER: {float(total_row['WER']):.6f}")
        report_lines.append(f"Recall: {float(total_row['Recall']):.6f}")
        report_lines.append(f"Precision: {float(total_row['Precision']):.6f}")
        report_lines.append(f"F1-Score: {float(total_row['F1-Score']):.6f}")
        report_lines.append("Top 5 frequent errors:")
        top_for_txt = top_freq_df.head(top_errors_report)
        if top_for_txt.empty:
            report_lines.append("1. (none)")
        else:
            for rank, row in enumerate(top_for_txt.itertuples(index=False), start=1):
                report_lines.append(
                    f"{rank}. {_safe_token(row.RefWord)} -> {_safe_token(row.HypWord)} ({int(row.Count)})"
                )
        report_lines.append("")

    summary_df = pd.DataFrame(summary_rows)
    if summary_tsv_path is None:
        summary_tsv_path = outputs_dir / "part_c_iterative_summary.tsv"
    summary_df.to_csv(summary_tsv_path, sep="\t", index=False, encoding="utf-8")

    if report_txt_path is None:
        report_txt_path = outputs_dir / "part_c_iterative_report.txt"
    Path(report_txt_path).write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    return summary_df, per_step_top_errors
