from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union


enable_debug = False
case_id = "0"
DEBUG_CASE_IDS = set() #{"043526"}


def extract_numbered_paragraphs(text: str, case_id: str = "Unknown") -> tuple:
    paragraphs = []
    missing_paragraphs = []
    marker_pattern = re.compile(r"\[(\d+)\]")

    matches = list(marker_pattern.finditer(text))
    if not matches:
        return paragraphs, missing_paragraphs

    marker_positions = []
    for m in matches:
        num_str = m.group(1)
        if len(num_str) < 4:
            marker_positions.append((int(num_str), m.start()))

    if not marker_positions:
        return paragraphs, missing_paragraphs

    first_num, first_pos = marker_positions[0]
    if first_num != 1 or first_pos > 0:
        preamble = text[:first_pos].strip()
        if preamble:
            paragraphs.append(f"[0]\n{preamble}")

    i = 0
    expected = 1
    last_position = len(text)
    reset_threshold = 10

    while i < len(marker_positions):
        num, pos = marker_positions[i]

        if num < expected - reset_threshold:
            expected = 1

        if num > expected:
            expected = num

        start = pos
        end = marker_positions[i + 1][1] if i + 1 < len(marker_positions) else last_position
        content = text[start:end].strip()

        if "-[" in content:
            parts = re.split(r"(\[\d+\]-\[\d+\])", content)
            main_content = parts[0].strip()
            if main_content:
                paragraphs.append(f"[{num}]\n{main_content}")
            for part in parts[1:]:
                if part:
                    paragraphs.append(f"[QUOTE {part}]")
        else:
            paragraphs.append(content)

        expected = num + 1
        i += 1

    return paragraphs, missing_paragraphs


def extract_marker(para: str) -> Union[int, None]:
    stripped = para.strip()
    match = re.match(r"\[(\d+)]", stripped)
    return int(match.group(1)) if match else None


def is_dual_language(paragraphs: List[str]) -> bool:
    global case_id

    if case_id == "015076":
        return False

    markers = [extract_marker(p) for p in paragraphs if extract_marker(p) is not None]
    if not markers:
        return False
    marker_counts = Counter(markers)
    num_repeats = sum(1 for count in marker_counts.values() if count > 1)
    return (num_repeats / len(marker_counts)) >= 0.90


def merge_single_language(paragraphs: List[str], case_num_paragraphs: int) -> List[str]:
    if not paragraphs:
        return []

    result = []
    i = 0
    n = len(paragraphs)

    while i < n:
        current_para = paragraphs[i]
        current_marker = extract_marker(current_para)
        if current_marker is not None:
            for missing in range(0, current_marker):
                result.append(f"[{missing}] <CONTENT_MISSING>")
            break
        i += 1

    if i == n:
        return ["[0] <CONTENT_MISSING>"] + paragraphs

    last_marker = extract_marker(paragraphs[i])
    current_para = paragraphs[i]
    i += 1

    while i < n:
        p = paragraphs[i]
        current_marker = extract_marker(p)

        if last_marker is None:
            current_para += " " + p
            i += 1
            continue

        expected = last_marker + 1

        if current_marker == expected:
            result.append(current_para)
            current_para = p
            last_marker = expected
            i += 1
        else:
            found_index = None
            found_target = None
            found_k = None

            lookahead_end = min(i + 5, n)
            for k in range(0, 6):
                target = expected + k
                for j in range(i, lookahead_end):
                    m = extract_marker(paragraphs[j])
                    if m == target:
                        found_index = j
                        found_target = target
                        found_k = k
                        break
                if found_index is not None:
                    break

            if found_index is not None:
                for j in range(i, found_index):
                    current_para += " " + paragraphs[j]

                if found_k == 0:
                    result.append(current_para)
                    current_para = paragraphs[found_index]
                    last_marker = found_target
                    i = found_index + 1
                else:
                    result.append(current_para)
                    for missing in range(expected, found_target):
                        result.append(f"[{missing}] <CONTENT_MISSING>")
                    current_para = paragraphs[found_index]
                    last_marker = found_target
                    i = found_index + 1
            else:
                current_para += " " + p
                i += 1

    result.append(current_para)
    return result


def merge_subparagraphs(
    paragraphs: List[str], case_num_paragraphs: int
) -> Tuple[List[str], Optional[int]]:
    global enable_debug, case_id

    if not paragraphs:
        return [], None

    dual_language = is_dual_language(paragraphs)

    if enable_debug:
        print(f"Dual Language: {dual_language}")
        print(f"Number of paragraphs: {len(paragraphs)}")
        markers = [extract_marker(p) for p in paragraphs]
        print(f"Paragraph markers: {markers}")

    if not dual_language:
        return merge_single_language(paragraphs, case_num_paragraphs), None

    midpoint = len(paragraphs) // 2
    max_offset = 5
    if case_id in ("051144"):
        max_offset = 2
    elif case_id in ("059126"):
        max_offset = 20

    split_index = None

    offsets = [0]
    for k in range(1, max_offset):
        offsets.extend([-k, k])
    search_indices = [midpoint + o for o in offsets if 0 <= midpoint + o < len(paragraphs)]

    for target_marker in range(1, 6):
        for i in search_indices:
            marker = extract_marker(paragraphs[i])
            if marker == target_marker:
                split_index = i
                break
        if split_index is not None:
            break

    if enable_debug:
        print(f"Final Split Index: {split_index}")

    if split_index is None:
        if enable_debug:
            print("split_index is None")
        return merge_single_language(paragraphs, case_num_paragraphs), None

    left = merge_single_language(paragraphs[:split_index], case_num_paragraphs)
    right = merge_single_language(paragraphs[split_index:], case_num_paragraphs)

    if enable_debug:
        print("----------- First Part ------------------")
        print(left)
        print("----------- Second Part ------------------")
        print(right)

    return left + right, split_index


def update_case_num_paragraphs(paragraphs: List[str], current_val: int) -> int:
    for para in reversed(paragraphs):
        mark = extract_marker(para)
        if mark is not None:
            return mark
    return current_val


def extract_case_info(file_path: Path) -> dict:
    global enable_debug, case_id

    content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    normalised_content = content.replace("\n", " ")
    case_id = file_path.stem

    enable_debug = case_id in DEBUG_CASE_IDS

    year_numbers = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", content)]
    year = str(max(year_numbers)) if year_numbers else "Unknown"

    judge_match = re.search(
        r"\[\d+\]\s*([A-Z][a-zA-ZÀ-ÿ'\.-]+,\s*(?:J\.|C\.J\.|A\.C\.J\.|J\.A\.|J\.T\.C\.C\.|J\.\s*\(ad hoc\)))",
        content,
    )
    judge = judge_match.group(1).strip() if judge_match else "Unknown"

    outcome_match = re.search(r"Application\s+(allowed|dismissed)", content, re.IGNORECASE)
    outcome = outcome_match.group(1).capitalize() if outcome_match else "Unknown"

    court_match = re.search(r"(Federal Court|Supreme Court|Court of Appeal|Appeal Division)", content)
    court = court_match.group(1) if court_match else "Unknown"

    acts = list(set(re.findall(r"\b([A-Z][A-Za-z\s]+? Act)\b", normalised_content)))
    regulations = list(set(re.findall(r"\b([A-Z][A-Za-z\s]+? Regulations?)\b", normalised_content)))
    citations = re.findall(r"\b\d{4}\s+[A-Z]{2,}\s+\d+\b", content)

    editor_match = re.search(r"Editor:\s*([^\n]+)", content)
    editor = editor_match.group(1).strip() if editor_match else "Unknown"

    paragraphs, missing_paragraphs = extract_numbered_paragraphs(content, case_id)

    def _get_marker_num(p: str) -> Optional[int]:
        m = re.match(r"\[(\d+)]", p.strip())
        return int(m.group(1)) if m else None

    marker_nums = [n for n in map(_get_marker_num, paragraphs) if n is not None]
    paragraphs_length = len(paragraphs)

    case_num_paragraphs = 0
    if marker_nums:
        last_marker = marker_nums[-1]
        if abs(last_marker - paragraphs_length) <= 6:
            case_num_paragraphs = last_marker
        else:
            case_num_paragraphs = marker_nums[0]
            prev = marker_nums[0]
            for n in marker_nums[1:]:
                if n <= prev and (prev - n) > 6:
                    break
                case_num_paragraphs = max(case_num_paragraphs, n)
                prev = n

            fallback_max = max(marker_nums) if marker_nums else 0
            if (
                fallback_max >= case_num_paragraphs + 10
                and fallback_max <= paragraphs_length + 5
            ):
                case_num_paragraphs = fallback_max

    paragraphs_length_original = len(paragraphs)
    dual_language = is_dual_language(paragraphs)

    paragraphs, split_idx = merge_subparagraphs(paragraphs, case_num_paragraphs)
    paragraphs_length = len(paragraphs)
    case_num_paragraphs = update_case_num_paragraphs(paragraphs, case_num_paragraphs)

    return {
        "case_id": case_id,
        "year": year,
        "judge": judge,
        "outcome": outcome,
        "court": court,
        "related_acts": acts,
        "related_regulations": regulations,
        "citations": citations,
        "editor": editor,
        "dual_lang": "Yes" if dual_language else "No",
        "paragraphs_length_full": paragraphs_length_original,
        "case_num_paragraphs": case_num_paragraphs,
        "paragraphs_length": paragraphs_length,
        "split_idx": split_idx,
        "missing_paragraphs": missing_paragraphs,
        "paragraphs": paragraphs,
        "full_text": content,
    }


def extract_metadata_from_cleaned_corpus(cleaned_cases_dir: Path, metadata_file: Path) -> List[dict]:
    cleaned_cases_dir = Path(cleaned_cases_dir)
    metadata_file = Path(metadata_file)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    case_records = [
        extract_case_info(fp)
        for fp in sorted(cleaned_cases_dir.glob("*.txt"))
    ]

    metadata_file.write_text(
        json.dumps(case_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return case_records
