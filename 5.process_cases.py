from __future__ import annotations

# Usage examples:
# cd /mnt/d/Workspace/COLIEE-2026
# python 5.process_cases.py \
#   --preset "${PRESET}" \
#   --start "${START_INDEX}" \
#   --num "${CASES_PER_TASK}"
#
# Local run example:
# python 5.process_cases.py --preset test_2026 --start 0 --num 20 --hf-cache-dir /tmp/hf_cache --model-id google/gemma-3-4b-it --model-name gemma-3-4b-it

import argparse
import os
from pathlib import Path

from config.paths import PRESETS
from utils.preprocessing.gemma_case_processing import (
    configure_hf_cache,
    print_gpu_status,
    run_case_enhancement_slice,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a slice of cases for legal-metadata extraction"
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="test_2026",
        help="Preset dataset configuration.",
    )
    parser.add_argument("--start", type=int, default=0, help="0-based start index.")
    parser.add_argument("--num", type=int, default=10, help="How many cases to process.")
    parser.add_argument("--offsit", type=int, default=0, help="Offset to skip N cases.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-3-27b-it",        
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma-3-27b-it",        
        help="Name suffix for saved case files.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Override Hugging Face cache directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]

    start_idx = args.offsit + args.start
    end_idx = start_idx + args.num

    cache_dir = args.hf_cache_dir or cfg.hf_cache_dir
    configure_hf_cache(cache_dir)

    print(f"🧮 start={start_idx}, end={end_idx - 1}, num={args.num}")
    print(f"📁 preset={args.preset}")
    print(f"📄 input metadata={cfg.metadata_en_file}")
    print(f"📦 enhanced output dir={cfg.output_enhanced_dir}")
    print(f"💾 hf cache dir={cache_dir}")

    print_gpu_status()

    enhanced_metadata_file = (
        cfg.output_enhanced_dir / f"enhanced_cases_{args.model_name.replace(':', '_')}.json"
    )

    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    run_case_enhancement_slice(
        en_metadata_file=cfg.metadata_en_file,
        output_enhanced_dir=cfg.output_enhanced_dir,
        enhanced_cases_dir=cfg.enhanced_cases_dir,
        enhanced_metadata_file=enhanced_metadata_file,
        model_id=args.model_id,
        model_name=args.model_name,
        cache_directory=cache_dir,
        start_idx=start_idx,
        end_idx=end_idx,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
