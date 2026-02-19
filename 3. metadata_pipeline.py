from __future__ import annotations

# Usage examples:
# cd /mnt/d/Workspace/COLIEE-2026
# python "3. metadata_pipeline.py" --preset train_2026
# python "3. metadata_pipeline.py" --preset test_2026
# python "3. metadata_pipeline.py" --preset combined_2026
#
# Or override paths explicitly, for example:
# python "3. metadata_pipeline.py" \
#   --cleaned-cases-dir 2026/Training/Dataset-Clean/CleanedCorpus \
#   --metadata-file 2026/Training/Dataset-Clean/processed_cases.json

import argparse
import json
from pathlib import Path

from config.paths import PRESETS, PipelineConfig
from utils.preprocessing.metadata_extraction import extract_metadata_from_cleaned_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run metadata extraction on deduplicated cleaned cases."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="train_2026",
        help="Preset dataset configuration.",
    )
    parser.add_argument(
        "--cleaned-cases-dir",
        type=Path,
        help="Override path to deduplicated cleaned cases directory.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        help="Override metadata output JSON file path.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[PipelineConfig, Path, Path]:
    cfg = PRESETS[args.preset]
    cleaned_cases_dir = args.cleaned_cases_dir or cfg.cleaned_cases_dir
    metadata_file = args.metadata_file or cfg.metadata_file
    return cfg, cleaned_cases_dir, metadata_file


def main() -> None:
    args = parse_args()
    cfg, cleaned_cases_dir, metadata_file = resolve_paths(args)

    if not cleaned_cases_dir.exists():
        raise FileNotFoundError(
            f"Cleaned cases directory not found: {cleaned_cases_dir}. "
            "Run deduplication pipeline first."
        )

    print(f"Preset: {args.preset}")
    print(f"Input cleaned cases dir: {cleaned_cases_dir}")
    print(f"Output metadata file: {metadata_file}")

    records = extract_metadata_from_cleaned_corpus(cleaned_cases_dir, metadata_file)

    report = {
        "preset": args.preset,
        "output_clean_dir": str(cfg.output_clean_dir),
        "cleaned_cases_dir": str(cleaned_cases_dir),
        "metadata_file": str(metadata_file),
        "num_cases_processed": len(records),
    }
    report_path = metadata_file.parent / "metadata_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"📦 Extracted metadata from {len(records)} cases -> saved to {metadata_file}"
    )
    print(f"📄 Metadata pipeline report saved to: {report_path}")


if __name__ == "__main__":
    main()
