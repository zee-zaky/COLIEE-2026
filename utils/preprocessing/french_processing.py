from __future__ import annotations

import json
import re
import inspect
import asyncio
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Set, Tuple

from utils.preprocessing.metadata_extraction import extract_marker
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from langdetect import DetectorFactory, detect
except Exception:  # pragma: no cover
    DetectorFactory = None
    detect = None

try:
    from googletrans import Translator
except Exception:  # pragma: no cover
    Translator = None


if DetectorFactory is not None:
    DetectorFactory.seed = 0


def load_cases(metadata_file: Path) -> List[dict]:
    return json.loads(Path(metadata_file).read_text(encoding="utf-8"))


def _iter_with_progress(items: List[dict], desc: str, show_progress: bool):
    if show_progress and tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def detect_french_paragraph_stats(cases: List[dict], show_progress: bool = True) -> List[dict]:
    if detect is None:
        raise ImportError("langdetect is required. Install with `pip install langdetect`.")

    results: List[dict] = []
    for case in _iter_with_progress(cases, "Detecting French paragraphs", show_progress):
        case_id = case.get("case_id", "Unknown")
        paragraphs = case.get("paragraphs", []) or []

        french_indices: List[int] = []
        for idx, para in enumerate(paragraphs):
            try:
                if detect(para.strip()) == "fr":
                    french_indices.append(idx)
            except Exception:
                continue

        french_count = len(french_indices)
        total = len(paragraphs)
        results.append(
            {
                "case_id": case_id,
                "num_paragraphs": total,
                "num_french_paragraphs": french_count,
                "percent_french": (french_count / total) if total else 0.0,
                "french_paragraph_indices": french_indices,
            }
        )
    return results


def summarize_french_stats(stats: List[dict]) -> dict:
    values = [row["num_french_paragraphs"] for row in stats]
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {"count": len(values), "min": min(values), "max": max(values), "mean": mean(values)}


def build_french_translation_review(
    french_stats: List[dict],
    processing_stats: List[dict],
) -> List[dict]:
    processing_map = {row["case_id"]: row for row in processing_stats}
    review_rows: List[dict] = []

    for row in french_stats:
        if row.get("num_french_paragraphs", 0) <= 0:
            continue

        case_id = row["case_id"]
        p = processing_map.get(case_id, {})
        num_translated = int(p.get("num_translated", 0))
        num_french = int(row.get("num_french_paragraphs", 0))
        num_deleted = int(p.get("num_deleted", 0))
        original_count = int(p.get("original_count", 0))
        num_final_paragraphs = int(p.get("num_final_paragraphs", 0))

        # Heuristic: deleted French paragraphs can be acceptable when dual-language
        # duplicates are removed and English counterparts remain.
        french_remaining_est = max(0, num_french - num_deleted)
        removed_by_dedup = num_deleted > 0 and num_final_paragraphs < original_count

        # Rerun signal: there are likely French paragraphs remaining, but none translated.
        needs_translation_rerun = (
            num_french > 0
            and num_translated == 0
            and not (removed_by_dedup and french_remaining_est == 0)
        )
        if needs_translation_rerun:
            rerun_reason = "french_remaining_untranslated"
        elif removed_by_dedup and french_remaining_est == 0:
            rerun_reason = "dual_language_dedup_kept_english"
        else:
            rerun_reason = "translation_not_required_or_completed"

        review_rows.append(
            {
                "case_id": case_id,
                "num_paragraphs": int(row.get("num_paragraphs", 0)),
                "num_french_paragraphs": num_french,
                "percent_french": float(row.get("percent_french", 0.0)),
                "french_paragraph_indices": row.get("french_paragraph_indices", []),
                "num_translated": num_translated,
                "num_deleted": num_deleted,
                "num_final_paragraphs": num_final_paragraphs,
                "french_remaining_est": french_remaining_est,
                "needs_translation_rerun": needs_translation_rerun,
                "rerun_reason": rerun_reason,
            }
        )

    review_rows.sort(
        key=lambda x: (x["needs_translation_rerun"], x["percent_french"], x["num_french_paragraphs"]),
        reverse=True,
    )
    return review_rows


def _normalize_fragment_suppressed(text: str) -> str:
    return re.sub(r"<fragment[^>]*>", "<FRAGMENT_SUPPRESSED>", text, flags=re.IGNORECASE)


def _resolve_maybe_awaitable(value):
    if not inspect.isawaitable(value):
        return value
    try:
        return asyncio.run(value)
    except RuntimeError:
        # Fallback for environments where an event loop is already present.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(value)
        finally:
            loop.close()


def _translate_text_with_status(
    text: str,
    translator: Optional[object],
    retries: int = 20,
) -> Tuple[str, str]:
    if translator is None:
        return text, "no_translator"

    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            result = translator.translate(text, src="fr", dest="en")
            result = _resolve_maybe_awaitable(result)
            translated = getattr(result, "text", None)
            if not isinstance(translated, str):
                translated = text
            translated = _normalize_fragment_suppressed(translated)
            if translated != text:
                return translated, f"translated_changed(attempt={attempt + 1})"
            if attempt < retries:
                time.sleep(0.2 * (attempt + 1))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.2 * (attempt + 1))

    if last_error:
        return text, f"translation_error({last_error})"
    return text, "translated_unchanged"


def _handle_para(para: str, is_french: bool, translator: Optional[object]) -> Tuple[str, str]:
    if "<CONTENT_MISSING>" in para:
        return para, "missing_content"
    if is_french:
        return _translate_text_with_status(para, translator)
    return para, "not_french"


def process_cases_with_french_handling(
    all_cases: List[dict],
    french_stats: List[dict],
    french_threshold_percent: float = 0.000001,
    show_progress: bool = True,
    debug_case_ids: Optional[Set[str]] = None,
) -> Tuple[List[dict], List[dict]]:
    translator = Translator() if Translator is not None else None
    debug_case_ids = debug_case_ids or set()

    french_stats_map: Dict[str, Set[int]] = {
        row["case_id"]: set(row["french_paragraph_indices"])
        for row in french_stats
        if row["percent_french"] > french_threshold_percent
    }

    processed_cases: List[dict] = []
    processing_stats: List[dict] = []

    for case in _iter_with_progress(all_cases, "Processing French-heavy cases", show_progress):
        case_id = case.get("case_id", "Unknown")
        dual_lang = case.get("dual_lang") == "Yes"
        paragraphs = case.get("paragraphs", []) or []
        french_indices = french_stats_map.get(case_id, set())

        para_dict: Dict[int, str] = {}
        lang_by_para_num: Dict[int, str] = {}
        translated_count = 0
        deleted_count = 0

        for idx, para in enumerate(paragraphs):
            para_num = extract_marker(para)
            if para_num is None:
                continue

            is_french = idx in french_indices
            is_missing = "<CONTENT_MISSING>" in para

            if para_num in para_dict:
                if dual_lang:
                    existing_para = para_dict[para_num]
                    existing_is_missing = "<CONTENT_MISSING>" in existing_para

                    if existing_is_missing and not is_missing:
                        translated_para, _ = _handle_para(para, is_french, translator)
                        para_dict[para_num] = translated_para
                        lang_by_para_num[para_num] = "fr" if is_french else "en"
                        translated_count += int(is_french)
                        deleted_count += 1
                    elif is_missing and not existing_is_missing:
                        deleted_count += 1
                    elif not is_missing and not existing_is_missing:
                        if not is_french and lang_by_para_num.get(para_num) == "fr":
                            para_dict[para_num] = para
                            lang_by_para_num[para_num] = "en"
                            translated_count = max(0, translated_count - 1)
                            deleted_count += 1
                        else:
                            deleted_count += 1
                    else:
                        deleted_count += 1
                else:
                    deleted_count += 1
                continue

            translated_para, translation_status = _handle_para(para, is_french, translator)
            para_dict[para_num] = translated_para
            lang_by_para_num[para_num] = "fr" if is_french else "en"
            if is_french:
                translated_count += 1
                if case_id in debug_case_ids:
                    print(
                        f"\n[DEBUG][{case_id}] idx={idx} marker=[{para_num}] "
                        f"lang=fr translated={'yes' if translated_para != para else 'no'} "
                        f"status={translation_status}"
                    )
                    print("[DEBUG] Original:")
                    print(para)
                    print("[DEBUG] Translated:")
                    print(translated_para)

        sorted_paragraphs = [para_dict[k] for k in sorted(para_dict)]
        updated_case = dict(case)
        updated_case["paragraphs"] = sorted_paragraphs
        processed_cases.append(updated_case)

        processing_stats.append(
            {
                "case_id": case_id,
                "original_count": len(paragraphs),
                "num_translated": translated_count,
                "num_deleted": deleted_count,
                "num_final_paragraphs": len(sorted_paragraphs),
            }
        )

    return processed_cases, processing_stats


def finalize_enriched_cases(cases: List[dict], show_progress: bool = True) -> List[dict]:
    updated_cases: List[dict] = []

    for case in _iter_with_progress(cases, "Finalizing enriched cases", show_progress):
        paragraphs = case.get("paragraphs", []) or []
        full_text_en = "\n\n".join(paragraphs)

        case = dict(case)
        case["full_text_en"] = full_text_en

        year_numbers = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", full_text_en)]
        case["year"] = str(max(year_numbers)) if year_numbers else "Unknown"

        last_marker = None
        for para in reversed(paragraphs):
            m = extract_marker(para)
            if m is not None:
                last_marker = m
                break
        case["case_num_paragraphs"] = last_marker if last_marker is not None else 0
        case["paragraphs_length"] = len(paragraphs)

        case.pop("decision_date", None)
        case.pop("split_idx", None)

        missing_nums = [
            extract_marker(p)
            for p in paragraphs
            if "<CONTENT_MISSING>" in p and extract_marker(p) is not None
        ]
        missing_nums = [m for m in missing_nums if m is not None]

        rebuilt = {}
        for k, v in case.items():
            rebuilt[k] = v
            if k == "missing_paragraphs":
                rebuilt["missing_paragraphs_num"] = len(missing_nums)
        if "missing_paragraphs" not in case:
            rebuilt["missing_paragraphs"] = missing_nums
            rebuilt["missing_paragraphs_num"] = len(missing_nums)

        updated_cases.append(rebuilt)

    return updated_cases


def write_json(path: Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
