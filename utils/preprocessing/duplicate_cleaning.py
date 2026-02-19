import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import re
import unicodedata


def count_fragments(text: str) -> int:
    """Count the number of <FRAGMENT_SUPPRESSED> masks in the text."""
    return text.count("<FRAGMENT_SUPPRESSED>")


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text: preserve accented characters, remove unwanted control chars."""
    # Normalize to NFC (e.g., é stays as one character)
    text = unicodedata.normalize("NFC", text)
    # Remove control characters except newline
    text = re.sub(r'[^\x20-\x7EÀ-ÿ\n]', '', text)
    return text


def remove_duplicate_files(input_dir: Path, output_dir: Path, excel_out: Path) -> pd.DataFrame:
    """
    Removes duplicate text files based on content hash.
    Adds size and <FRAGMENT_SUPPRESSED> mask counts for both original and duplicate.
    Only writes unique documents to output_dir.
    """
    output_dir.mkdir(exist_ok=True)
    hashes = {}
    duplicate_pairs = []

    for file_path in input_dir.glob("*.txt"):
        raw_content = file_path.read_text(encoding='utf-8', errors='ignore')
        content = normalize_unicode(raw_content)
        file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        size = len(content)
        num_masks = count_fragments(content)

        if file_hash in hashes:
            original_name = hashes[file_hash]
            original_path = input_dir / original_name
            original_content = normalize_unicode(original_path.read_text(encoding='utf-8', errors='ignore'))

            duplicate_pairs.append({
                "Original": original_name,
                "Duplicate": file_path.name,
                "Original Size": len(original_content),
                "Duplicate Size": size,
                "Original <FRAGMENT_SUPPRESSED> Count": count_fragments(original_content),
                "Duplicate <FRAGMENT_SUPPRESSED> Count": num_masks,
            })
            continue

        # Write only unique content (cleaned and normalized)
        hashes[file_hash] = file_path.name
        clean_text = re.sub(r'\n{2,}', '\n', content.strip())
        (output_dir / file_path.name).write_text(clean_text, encoding='utf-8')

    # Ensure DataFrame always has the expected columns even if empty
    columns = [
        "Original",
        "Duplicate",
        "Original Size",
        "Duplicate Size",
        "Original <FRAGMENT_SUPPRESSED> Count",
        "Duplicate <FRAGMENT_SUPPRESSED> Count"
    ]
    df_duplicates = pd.DataFrame(duplicate_pairs, columns=columns)
    df_duplicates.to_excel(excel_out, index=False)

    print(f"🔁 {len(df_duplicates)} duplicate files detected.") 
    print(f"🧾 Full duplicate list saved to: {excel_out}")

    return df_duplicates


def remove_duplicate_labels(label_file: Path, duplicate_map: dict, output_dir: Path) -> dict:
    """
    Load label JSON, replace duplicate case IDs in both query and citation positions,
    deduplicate all, and save to output_dir as cleaned_labels.json.
    
    Args:
        label_file (Path): Path to the original labels JSON file.
        duplicate_map (dict): Dictionary mapping duplicate IDs to original IDs.
        output_dir (Path): Directory to save cleaned_labels.json.

    Returns:
        dict: Cleaned labels with remapped and deduplicated IDs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cleaned_labels.json"

    with open(label_file, "r", encoding="utf-8") as f:
        labels = json.load(f)

    cleaned = defaultdict(set)
    remapped_queries = 0
    remapped_citations = 0

    for qid, cids in labels.items():
        mapped_qid = duplicate_map.get(qid, qid)
        if mapped_qid != qid:
            remapped_queries += 1
        for cid in cids:
            mapped_cid = duplicate_map.get(cid, cid)
            if mapped_cid != cid:
                remapped_citations += 1
            cleaned[mapped_qid].add(mapped_cid)

    # Convert sets to sorted lists for JSON compatibility
    cleaned_final = {qid: sorted(cids) for qid, cids in cleaned.items()}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_final, f, indent=2)

    print(f"✅ Cleaned and mapped labels saved to: {output_file}")
    print(f"🔁 Replaced {remapped_queries} duplicate query IDs and {remapped_citations} citation IDs.")
    print(f"📊 Total cleaned queries: {len(cleaned_final)}")

    return cleaned_final


def verify_labels_integrity(cleaned_labels_path: Path, cleaned_cases_dir: Path) -> list:
    """
    Verify that all query and citation case IDs in cleaned_labels.json
    correspond to files in the cleaned_cases_dir.
    
    Returns a sorted list of any missing case IDs.
    """
    with open(cleaned_labels_path, "r", encoding="utf-8") as f:
        cleaned_labels = json.load(f)

    # Collect both query IDs and all cited case IDs
    cited_ids = {cid.replace(".txt", "") for cids in cleaned_labels.values() for cid in cids}
    query_ids = {qid.replace(".txt", "") for qid in cleaned_labels.keys()}
    all_ids = query_ids.union(cited_ids)

    # File stems from filesystem
    existing_files = {fp.stem for fp in cleaned_cases_dir.glob("*.txt")}
    missing = sorted(all_ids - existing_files)

    if missing:
        print(f"❌ {len(missing)} case files are missing from CleanedCorpus:")
        print(missing[:10], "..." if len(missing) > 10 else "")
    else:
        print("✅ All query and citation case files are present in CleanedCorpus.")

    return missing


def check_duplicate_query_labels(labels_path: Path):
    """
    Check cleaned_labels.json for:
    1. Duplicate citations in each query's list.
    2. Query ID present in its own citation list.
    3. Duplicate query IDs in the JSON file (not just dict keys).
    """
    # Load raw JSON text to detect duplicate query keys
    raw_text = labels_path.read_text(encoding="utf-8")
    query_keys = re.findall(r'"(\d{6}\.txt)"\s*:', raw_text)
    query_counter = Counter(query_keys)
    duplicate_query_ids = [qid for qid, count in query_counter.items() if count > 1]

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    duplicates_found = defaultdict(list)
    self_citations = []

    for qid, cids in labels.items():
        seen = set()
        for cid in cids:
            if cid in seen:
                duplicates_found[qid].append(cid)
            else:
                seen.add(cid)
        if qid in cids:
            self_citations.append(qid)

    # Report duplicate citations
    if duplicates_found:
        print(f"❌ Duplicate citations found in {len(duplicates_found)} queries:")
        for qid, dups in duplicates_found.items():
            print(f"  - {qid}: {dups}")
    else:
        print("✅ No duplicate citations found in any query.")

    # Report self-citations
    if self_citations:
        print(f"\n❌ Query label appears in its own citations in {len(self_citations)} cases:")
        for qid in self_citations:
            print(f"  - {qid}")
    else:
        print("✅ No queries cite themselves.")

    # Report duplicate query IDs
    if duplicate_query_ids:
        print(f"\n❌ Duplicate query IDs found in the file: {len(duplicate_query_ids)}")
        for qid in duplicate_query_ids:
            print(f"  - {qid}")
    else:
        print("✅ No duplicate query IDs found in the file.")

    return {
        "duplicates": dict(duplicates_found),
        "self_citations": self_citations,
        "duplicate_queries": duplicate_query_ids
    }
