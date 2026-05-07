import argparse
from pathlib import Path

import pandas as pd

from asr_project.config import ProjectConfig
from asr_project.stage_c_iterative import build_default_stage_c_steps, run_stage_c_iterative


def _resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path


def main():
    parser = argparse.ArgumentParser(description="Run Stage C iterative normalization analysis.")
    parser.add_argument(
        "--transcriptions_tsv",
        default="data/outputs/part_a_transcriptions.tsv",
        help="Path to Part A transcription TSV.",
    )
    parser.add_argument(
        "--outputs_dir",
        default="data/outputs",
        help="Directory for Stage C output files.",
    )
    parser.add_argument(
        "--frequent_errors_min_count",
        type=int,
        default=2,
        help="Minimum count threshold for frequent errors.",
    )
    parser.add_argument(
        "--top_errors_eval",
        type=int,
        default=10,
        help="Number of top errors to save per stage.",
    )
    parser.add_argument(
        "--top_errors_report",
        type=int,
        default=5,
        help="Number of top errors to include in the TXT report.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    cfg = ProjectConfig.default(project_root)
    cfg.ensure_dirs()

    transcriptions_tsv = _resolve_path(args.transcriptions_tsv, project_root)
    outputs_dir = _resolve_path(args.outputs_dir, project_root)
    summary_tsv = outputs_dir / "part_c_iterative_summary.tsv"
    report_txt = outputs_dir / "part_c_iterative_report.txt"

    summary_df, _ = run_stage_c_iterative(
        transcriptions_tsv=transcriptions_tsv,
        outputs_dir=outputs_dir,
        steps=build_default_stage_c_steps(),
        frequent_errors_min_count=args.frequent_errors_min_count,
        top_errors_eval=args.top_errors_eval,
        top_errors_report=args.top_errors_report,
        summary_tsv_path=summary_tsv,
        report_txt_path=report_txt,
    )

    pd.set_option("display.max_colwidth", 120)
    print(summary_df.to_string(index=False))
    print()
    print(f"Wrote summary TSV: {summary_tsv}")
    print(f"Wrote report TXT: {report_txt}")


if __name__ == "__main__":
    main()
