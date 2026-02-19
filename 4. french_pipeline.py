from __future__ import annotations

# Usage examples:
# cd /mnt/d/Workspace/COLIEE-2026
# python "4. french_pipeline.py" --preset train_2026
# python "4. french_pipeline.py" --preset test_2026
# python "4. french_pipeline.py" --preset combined_2026
# python "4. french_pipeline.py" --preset test_2026 \
#   --debug-case-id 004038 \
#   --debug-case-id 012345


import argparse
import json
from pathlib import Path

from config.paths import PRESETS
from utils.preprocessing.french_processing import (
    build_french_translation_review,
    detect_french_paragraph_stats,
    finalize_enriched_cases,
    load_cases,
    process_cases_with_french_handling,
    summarize_french_stats,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and translate French paragraphs in extracted metadata."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="train_2026",
        help="Preset dataset configuration.",
    )
    parser.add_argument("--metadata-file", type=Path, help="Override input metadata JSON.")
    parser.add_argument("--metadata-en-file", type=Path, help="Override output EN metadata JSON.")
    parser.add_argument("--french-stats-file", type=Path, help="Override French stats JSON.")
    parser.add_argument(
        "--french-processing-stats-file",
        type=Path,
        help="Override processing stats JSON.",
    )
    parser.add_argument(
        "--french-translation-review-file",
        type=Path,
        help="Override French-only translation review JSON.",
    )
    parser.add_argument(
        "--french-threshold-percent",
        type=float,
        default=0.000001,
        help="Minimum fraction of French paragraphs to mark a case as French-containing.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--debug-case-id",
        action="append",
        default=[],
        help="Case ID to print original/translated French paragraphs for. Repeat flag for multiple IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]

    metadata_file = args.metadata_file or cfg.metadata_file
    metadata_en_file = args.metadata_en_file or cfg.metadata_en_file
    french_stats_file = args.french_stats_file or cfg.french_stats_file
    french_processing_stats_file = (
        args.french_processing_stats_file or cfg.french_processing_stats_file
    )
    french_translation_review_file = (
        args.french_translation_review_file or cfg.french_translation_review_file
    )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}. Run `3. metadata_pipeline.py` first."
        )

    print(f"Preset: {args.preset}")
    print(f"Input metadata file: {metadata_file}")
    print(f"Output EN metadata file: {metadata_en_file}")
    print(f"French stats file: {french_stats_file}")
    print(f"French processing stats file: {french_processing_stats_file}")
    print(f"French translation review file: {french_translation_review_file}")

    cases = load_cases(metadata_file)
    show_progress = not args.no_progress
    french_stats = detect_french_paragraph_stats(cases, show_progress=show_progress)
    write_json(french_stats_file, french_stats)

    summary = summarize_french_stats(french_stats)
    print(
        "French paragraph summary: "
        f"count={summary['count']} min={summary['min']} "
        f"max={summary['max']} mean={summary['mean']:.2f}"
    )

    processed_cases, processing_stats = process_cases_with_french_handling(
        cases,
        french_stats,
        french_threshold_percent=args.french_threshold_percent,
        show_progress=show_progress,
        debug_case_ids=set(args.debug_case_id),
    )
    write_json(french_processing_stats_file, processing_stats)
    french_translation_review = build_french_translation_review(french_stats, processing_stats)
    write_json(french_translation_review_file, french_translation_review)

    updated_cases = finalize_enriched_cases(processed_cases, show_progress=show_progress)
    write_json(metadata_en_file, updated_cases)

    report = {
        "preset": args.preset,
        "metadata_file": str(metadata_file),
        "metadata_en_file": str(metadata_en_file),
        "french_stats_file": str(french_stats_file),
        "french_processing_stats_file": str(french_processing_stats_file),
        "french_translation_review_file": str(french_translation_review_file),
        "num_cases": len(cases),
        "num_cases_with_any_french": sum(1 for r in french_stats if r["num_french_paragraphs"] > 0),
        "num_cases_translated_or_deduped": sum(
            1 for r in processing_stats if (r["num_translated"] > 0 or r["num_deleted"] > 0)
        ),
        "num_cases_needing_translation_rerun": sum(
            1 for r in french_translation_review if r["needs_translation_rerun"]
        ),
    }
    report_path = metadata_en_file.parent / "french_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"✅ Saved EN metadata to {metadata_en_file}")
    print(
        "🔎 Saved French-only translation review to "
        f"{french_translation_review_file} "
        f"(needs rerun: {report['num_cases_needing_translation_rerun']})"
    )
    print(f"📄 French pipeline report saved to: {report_path}")


if __name__ == "__main__":
    main()
