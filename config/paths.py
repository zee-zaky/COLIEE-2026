from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


DEFAULT_OUTPUT_CLEAN_DIR = "Dataset-Clean"
DEFAULT_CLEANED_CASES_DIR = f"{DEFAULT_OUTPUT_CLEAN_DIR}/CleanedCorpus"
DEFAULT_METADATA_FILE = f"{DEFAULT_OUTPUT_CLEAN_DIR}/processed_cases.json"
DEFAULT_METADATA_EN_FILE = f"{DEFAULT_OUTPUT_CLEAN_DIR}/processed_cases_en.json"
DEFAULT_FRENCH_STATS_FILE = f"{DEFAULT_OUTPUT_CLEAN_DIR}/french_paragraph_stats.json"
DEFAULT_FRENCH_PROCESSING_STATS_FILE = (
    f"{DEFAULT_OUTPUT_CLEAN_DIR}/french_processing_stats.json"
)
DEFAULT_FRENCH_TRANSLATION_REVIEW_FILE = (
    f"{DEFAULT_OUTPUT_CLEAN_DIR}/french_translation_review.json"
)
DEFAULT_OUTPUT_ENHANCED_DIR = "Dataset-Enhanced"
DEFAULT_ENHANCED_CASES_DIR = f"{DEFAULT_OUTPUT_ENHANCED_DIR}/Cases"
DEFAULT_ENHANCED_LOGS_DIR = f"{DEFAULT_OUTPUT_ENHANCED_DIR}/logs"
DEFAULT_MA_METADATA_DIR = f"{DEFAULT_OUTPUT_ENHANCED_DIR}/MA_enhanced_cases"
DEFAULT_ENHANCED_METADATA_PREFIX = "enhanced_cases"
DEFAULT_HF_CACHE_DIR = "/nesi/nobackup/uoa04665/mzak071/hf_cache"


@dataclass(frozen=True)
class PipelineConfig:
    base_dir: Path
    cases_dir: Path
    labels_file: Optional[Path]
    output_clean_dir: Path
    cleaned_cases_dir: Path
    metadata_file: Path
    metadata_en_file: Path
    french_stats_file: Path
    french_processing_stats_file: Path
    french_translation_review_file: Path
    output_enhanced_dir: Path
    enhanced_cases_dir: Path
    enhanced_logs_dir: Path
    ma_metadata_dir: Path
    enhanced_metadata_prefix: str
    hf_cache_dir: Path
    no_labels_file: Optional[Path] = None


def _preset(
    *,
    base_dir: str,
    cases_dir: str,
    labels_file: Optional[str],
    output_clean_dir: str = DEFAULT_OUTPUT_CLEAN_DIR,
    cleaned_cases_dir: str = DEFAULT_CLEANED_CASES_DIR,
    metadata_file: str = DEFAULT_METADATA_FILE,
    metadata_en_file: str = DEFAULT_METADATA_EN_FILE,
    french_stats_file: str = DEFAULT_FRENCH_STATS_FILE,
    french_processing_stats_file: str = DEFAULT_FRENCH_PROCESSING_STATS_FILE,
    french_translation_review_file: str = DEFAULT_FRENCH_TRANSLATION_REVIEW_FILE,
    output_enhanced_dir: str = DEFAULT_OUTPUT_ENHANCED_DIR,
    enhanced_cases_dir: str = DEFAULT_ENHANCED_CASES_DIR,
    enhanced_logs_dir: str = DEFAULT_ENHANCED_LOGS_DIR,
    ma_metadata_dir: str = DEFAULT_MA_METADATA_DIR,
    enhanced_metadata_prefix: str = DEFAULT_ENHANCED_METADATA_PREFIX,
    hf_cache_dir: str = DEFAULT_HF_CACHE_DIR,
    no_labels_file: Optional[str] = None,
) -> PipelineConfig:
    base = Path(base_dir)
    hf_cache = Path(hf_cache_dir)
    if not hf_cache.is_absolute():
        hf_cache = base / hf_cache_dir
    return PipelineConfig(
        base_dir=base,
        cases_dir=base / cases_dir,
        labels_file=(base / labels_file) if labels_file else None,
        output_clean_dir=base / output_clean_dir,
        cleaned_cases_dir=base / cleaned_cases_dir,
        metadata_file=base / metadata_file,
        metadata_en_file=base / metadata_en_file,
        french_stats_file=base / french_stats_file,
        french_processing_stats_file=base / french_processing_stats_file,
        french_translation_review_file=base / french_translation_review_file,
        output_enhanced_dir=base / output_enhanced_dir,
        enhanced_cases_dir=base / enhanced_cases_dir,
        enhanced_logs_dir=base / enhanced_logs_dir,
        ma_metadata_dir=base / ma_metadata_dir,
        enhanced_metadata_prefix=enhanced_metadata_prefix,
        hf_cache_dir=hf_cache,
        no_labels_file=(base / no_labels_file) if no_labels_file else None,
    )


PRESETS: Dict[str, PipelineConfig] = {
    "train_2025": _preset(
        base_dir="2025/Training",
        cases_dir="Dataset/cases",
        labels_file="Dataset/task1_train_labels_2025.json",
        no_labels_file=None,
    ),
    "test_2025": _preset(
        base_dir="2025/Test",
        cases_dir="Dataset/cases",
        labels_file="Dataset/task1_test_labels_2025.json",
        no_labels_file="Dataset/task1_test_no_labels_2025.json",
    ),
    "train_2026": _preset(
        base_dir="2026/Training",
        cases_dir="Dataset/cases",
        labels_file="Dataset/clean_task1_train_labels_2026.json",
        no_labels_file=None,
    ),
    "test_2026": _preset(
        base_dir="2026/Test",
        cases_dir="Dataset/cases",
        labels_file=None,
        no_labels_file="Dataset/task1_test_no_labels_2026.json",
    ),
    "combined_2026": _preset(
        base_dir="2026/Combined",
        cases_dir="Dataset/cases",
        labels_file=None,
        no_labels_file=None,
    ),
}
