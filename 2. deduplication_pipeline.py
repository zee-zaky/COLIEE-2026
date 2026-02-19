from __future__ import annotations

# Usage examples:
# cd /mnt/d/Workspace/COLIEE-2026
# python "2. deduplication_pipeline.py" --preset train_2026
# python "2. deduplication_pipeline.py" --preset test_2026
# python "2. deduplication_pipeline.py" --preset combined_2026
#
# Or override paths explicitly, for example:
# python utils/preprocessing/preprocessing_pipeline.py \
#   --cases-dir Training/Dataset/cases \
#   --labels-file Training/Dataset/clean_task1_train_labels_2026.json \
#   --output-base-dir Training/Dataset-2026-Clean

import argparse
import json
from pathlib import Path

from config.paths import PRESETS, PipelineConfig


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    Allows selecting a preset or overriding individual path arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run duplicate cleaning + label cleaning pipeline."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="train_2026",
        help="Preset dataset configuration.",
    )
    parser.add_argument("--cases-dir", type=Path, help="Override path to case .txt files.")
    parser.add_argument("--labels-file", type=Path, help="Override path to labels JSON.")
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        help="Override output base directory.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> PipelineConfig:
    """
    Build the effective PipelineConfig from the selected preset and any overrides.
    """
    cfg = PRESETS[args.preset]
    output_clean_dir = args.output_base_dir or cfg.output_clean_dir
    return PipelineConfig(
        base_dir=cfg.base_dir,
        cases_dir=args.cases_dir or cfg.cases_dir,
        labels_file=args.labels_file if args.labels_file is not None else cfg.labels_file,
        output_clean_dir=output_clean_dir,
        cleaned_cases_dir=(
            output_clean_dir / cfg.cleaned_cases_dir.name
            if args.output_base_dir is not None
            else cfg.cleaned_cases_dir
        ),
        metadata_file=(
            output_clean_dir / cfg.metadata_file.name
            if args.output_base_dir is not None
            else cfg.metadata_file
        ),
        metadata_en_file=(
            output_clean_dir / cfg.metadata_en_file.name
            if args.output_base_dir is not None
            else cfg.metadata_en_file
        ),
        french_stats_file=(
            output_clean_dir / cfg.french_stats_file.name
            if args.output_base_dir is not None
            else cfg.french_stats_file
        ),
        french_processing_stats_file=(
            output_clean_dir / cfg.french_processing_stats_file.name
            if args.output_base_dir is not None
            else cfg.french_processing_stats_file
        ),
        french_translation_review_file=(
            output_clean_dir / cfg.french_translation_review_file.name
            if args.output_base_dir is not None
            else cfg.french_translation_review_file
        ),
        output_enhanced_dir=cfg.output_enhanced_dir,
        enhanced_cases_dir=cfg.enhanced_cases_dir,
        enhanced_logs_dir=cfg.enhanced_logs_dir,
        ma_metadata_dir=cfg.ma_metadata_dir,
        hf_cache_dir=cfg.hf_cache_dir,
        no_labels_file=cfg.no_labels_file,
    )


def main() -> None:
    """
    Main pipeline orchestration:
    - Parse arguments and resolve configuration
    - Run duplicate file detection and removal
    - Remove duplicate labels based on file dedup map
    - Verify label/case integrity and check for duplicate queries in labels
    - Write a JSON report summarizing outputs
    """
    args = parse_args()
    cfg = resolve_config(args)

    # Import local module that implements cleaning helpers.
    from utils.preprocessing.duplicate_cleaning import (
        check_duplicate_query_labels,
        remove_duplicate_files,
        remove_duplicate_labels,
        verify_labels_integrity,
    )

    # Prepare output paths inside the output base dir.
    output_cases_dir = cfg.cleaned_cases_dir
    duplicate_excel_file = cfg.output_clean_dir / "duplicate_cases.xlsx"
    cleaned_labels_path = cfg.output_clean_dir / "cleaned_labels.json"
    report_path = cfg.output_clean_dir / "pipeline_report.json"

    # Ensure output directories exist.
    cfg.output_clean_dir.mkdir(parents=True, exist_ok=True)
    output_cases_dir.mkdir(parents=True, exist_ok=True)

    # Log resolved paths for user visibility.
    print(f"Cases dir: {cfg.cases_dir}")
    print(f"Labels file: {cfg.labels_file}")
    print(f"Output clean dir: {cfg.output_clean_dir}")

    # Detect and remove duplicate case files. Function returns a DataFrame
    # mapping duplicates to their canonical originals and writes an Excel report.
    df_duplicates = remove_duplicate_files(
        input_dir=cfg.cases_dir,
        output_dir=output_cases_dir,
        excel_out=duplicate_excel_file,
    )
    # Build a mapping from duplicate filename -> original filename for label cleanup.
    duplicate_map = dict(zip(df_duplicates["Duplicate"], df_duplicates["Original"]))

    labels_processed = False
    missing_files = []
    duplicate_label_report = {
        "duplicates": {},
        "self_citations": [],
        "duplicate_queries": [],
    }

    if cfg.labels_file is not None:
        if not cfg.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {cfg.labels_file}")

        # Remove/merge duplicate labels according to the duplicate_map and write cleaned labels.
        remove_duplicate_labels(
            label_file=cfg.labels_file,
            duplicate_map=duplicate_map,
            output_dir=cfg.output_clean_dir,
        )

        # Verify that every label has a corresponding case file and collect missing files.
        missing_files = verify_labels_integrity(
            cleaned_labels_path=cleaned_labels_path,
            cleaned_cases_dir=output_cases_dir,
        )
        # Check for duplicate query texts inside the cleaned labels JSON.
        duplicate_label_report = check_duplicate_query_labels(cleaned_labels_path)
        labels_processed = True
    else:
        print("ℹ️ No labels file configured. Skipping label cleanup and validation steps.")

    # Compose a small report summarizing the pipeline results and write it to disk.
    report = {
        "preset": args.preset,
        "cases_dir": str(cfg.cases_dir),
        "labels_file": str(cfg.labels_file) if cfg.labels_file is not None else None,
        "labels_processed": labels_processed,
        "output_clean_dir": str(cfg.output_clean_dir),
        "metadata_file": str(cfg.metadata_file),
        "cleaned_cases_dir": str(output_cases_dir),
        "duplicate_excel_file": str(duplicate_excel_file),
        "cleaned_labels_path": str(cleaned_labels_path) if labels_processed else None,
        "num_duplicates": int(len(df_duplicates)),
        "num_missing_case_files": int(len(missing_files)),
        "missing_case_files": missing_files,
        "duplicate_label_report": duplicate_label_report,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"📄 Pipeline report saved to: {report_path}")


if __name__ == "__main__":
    main()
