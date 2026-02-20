from __future__ import annotations

import ast
import gc
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from huggingface_hub import login
except Exception:  # pragma: no cover
    login = None

try:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
except Exception:  # pragma: no cover
    AutoProcessor = None
    Gemma3ForConditionalGeneration = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


ASPECT_SPECS: List[Tuple[str, str, str]] = [
    (
        "statutes",
        "[list of extracted statutes from the case text (including mentioned year and sections in one item together)]",
        "Extract DISTINCT statutes (laws) or subordinate instrument  (containing the word act / regulations / rules ... etc) that is **explicitly named**  and mentioned in the case text",
    ),
    (
        "legal_issues",
        "['list of legal issues addressed in the case, stated in an abstracted and generalised manner']",
        "Identify and list the core legal issues addressed in the case. Phrase each issue abstractly, without using party names or case-specific details. Focus on the legal questions the court needed to resolve, suitable for metadata extraction.",
    ),
    (
        "legal_topics",
        "['list of broad doctrinal labels, not case-specific names']",
        "Identify and list the overarching legal doctrines or thematic areas addressed in the case. Use broad, abstract terms. Avoid using party names, statute names, or case-specific phrasing. Focus on categorising the legal reasoning into general doctrinal areas suitable for metadata extraction.",
    ),
    (
        "factual_background",
        "['list of items summarising the factual background of the case in an abstract, generalised manner']",
        "Extract only the factual background from the case text. Summarise the relevant events, actors, and circumstances that gave rise to the legal dispute. Avoid legal analysis, arguments, or conclusions. Write in an abstracted, case-neutral tone suitable for metadata extraction.",
    ),
    (
        "key_arguments",
        """{
                "Applicants": ["list of items covering all arguments presented by the applicant(s)"],
                "Respondents": ["list of items covering all arguments presented by the respondent(s)"]
            }""",
        "Extract and summarise the key legal and factual arguments made by each side in the case. Separate the arguments clearly by party. Avoid paraphrasing the court’s evaluation or findings—focus strictly on what each party claimed or contested.",
    ),
    (
        "court_analysis",
        "['list of items explaining what the court found and *why and how* it reached its decision']",
        "Summarise the court’s reasoning and analysis. Focus on the key findings and the legal or factual rationale behind the decision. Explain how the court interpreted statutes, weighed evidence, or applied legal principles to resolve the issues. Avoid repeating party arguments—focus on the court’s own logic and conclusions.",
    ),
    (
        "final_outcome",
        "['list of outcomes or orders issued by the court']",
        "Identify and list the final outcome(s) or orders made by the court. Focus on the court’s binding decisions—e.g., whether the application was granted or dismissed, any declaratory or injunctive relief, cost orders, or directions to parties. Use neutral and formal phrasing. Do not include reasoning or background—only the court's final determinations suitable for metadata extraction.",
    ),
]


def configure_hf_cache(cache_dir: Path) -> None:
    cache = str(cache_dir)
    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = cache
    os.environ["TRANSFORMERS_CACHE"] = cache


def build_prompt(aspect_key: str, format_snippet: str, instruction: str) -> str:
    return f"""You are a legal metadata expert. {instruction} and return:
{{{{
  "{aspect_key}": {format_snippet}
}}}}
(no extra keys or commentary)

=======================
CASE TEXT STARTS
=======================
{{TEXT}}
=======================
CASE TEXT ENDS
=======================
Remember, you need to {instruction} and return the information in a structured JSON object with the following format:
{{{{
  "{aspect_key}": {format_snippet}
}}}}
(no extra keys or commentary)
"""

# Create PROMPTS dict automatically
PROMPTS: Dict[str, str] = {
    aspect_key: build_prompt(aspect_key, format_snippet, instruction)
    for (aspect_key, format_snippet, instruction) in ASPECT_SPECS
}


def setup_logger(case_id: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case_id}.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def print_gpu_status() -> None:
    if torch is None:
        print("⚠️ torch is not installed; skipping GPU status.")
        return
    print("🖥️  GPU status via nvidia-smi:")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ nvidia-smi failed: {e}")

    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.mem_get_info(i)
        print(
            f"GPU {i}: free {mem[0] // (1024**2)} MiB / total {mem[1] // (1024**2)} MiB",
            flush=True,
        )
        print(torch.cuda.memory_summary(device=i), flush=True)


def load_model(model_id: str, cache_directory: Path):
    if torch is None or AutoProcessor is None or Gemma3ForConditionalGeneration is None:
        raise ImportError(
            "Missing dependencies for model load. Install torch + transformers in runtime env."
        )
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total_mem = torch.cuda.get_device_properties(i).total_memory
        max_memory[i] = int(total_mem * 0.6)
    max_memory["cpu"] = 200 * 1024**3
    print(max_memory)

    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=str(cache_directory),
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            top_k=None,
            top_p=None,
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        return model, processor
    except OSError as e:
        msg = str(e)
        if "gated repo" in msg.lower() or "401" in msg:
            raise RuntimeError(
                "Cannot access gated Hugging Face model. "
                "Ensure your account has accepted model access and provide token via "
                "`--hf-token` or env `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`."
            ) from e
        raise
    except RuntimeError as e:
        print(f"❌ Model loading failed: {e}")
        raise


def query_aspect(
    aspect: str,
    tmpl: str,
    text: str,
    model,
    processor,
) -> dict:
    prompt = tmpl.format(TEXT=text)
    logging.info(
        "#────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────#"
    )
    logging.info(f"\n\n🟦 Aspect: {aspect}\n--- Prompt ---\n{prompt}")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a legal metadata expert."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            top_p=None,
            top_k=None,
        )
        output_ids = output[0][input_len:]

    raw = processor.decode(output_ids, skip_special_tokens=True)
    logging.info(f"--- Raw output for `{aspect}` ---\n{raw}")

    first, last = raw.find("{"), raw.rfind("}")
    if first == -1 or last == -1:
        raise ValueError(f"Couldn’t find JSON in output for `{aspect}`:\n{raw}")
    json_str = raw[first : last + 1]

    try:
        parsed = json.loads(json_str)
        logging.info(f"--- Parsed JSON for `{aspect}` ---\n{json.dumps(parsed, indent=2)}")
        return parsed
    except json.JSONDecodeError as e:
        logging.warning(f"Standard JSON parsing failed: {e}\nTrying ast.literal_eval fallback...")
        try:
            parsed = ast.literal_eval(json_str)
            logging.info(f"--- Parsed with ast.literal_eval for `{aspect}` ---\n{parsed}")
            return parsed
        except Exception as ast_e:
            raise ValueError(
                f"❌ Fallback parsing also failed for `{aspect}`: {ast_e}\n\nRaw:\n{raw}"
            )

def _normalize_key(k: str) -> str:
    # strips whitespace and also strips wrapping quotes if the model included them as text
    return k.strip().strip('"').strip("'")

def process_case(text: str, case_id: str, model, processor, output_enhanced_dir: Path) -> dict:
    log_dir = output_enhanced_dir / "logs"
    setup_logger(case_id, log_dir)
    logging.info(f"\n\n🗂️ Processing case ID: {case_id}")

    aggregated = {}
    for aspect, tmpl in PROMPTS.items():        
        piece = query_aspect(aspect, tmpl, text, model, processor)

        # normalize keys
        if isinstance(piece, dict):
            piece_norm = {_normalize_key(k): v for k, v in piece.items()}
        else:
            raise ValueError(f"Model returned non-dict for {aspect}: {type(piece)}")
        
        if aspect == "key_arguments":
            if "key_arguments" not in piece_norm:
                raise KeyError(f"Missing key_arguments in output keys={list(piece_norm.keys())}")
            aggregated["key_arguments"] = piece_norm["key_arguments"]
        else:
            if aspect not in piece_norm:
                raise KeyError(f"Missing {aspect} in output keys={list(piece_norm.keys())}")
            aggregated[aspect] = piece_norm[aspect]

    return aggregated


def run_case_enhancement_slice(
    *,
    en_metadata_file: Path,
    output_enhanced_dir: Path,
    enhanced_cases_dir: Path,
    enhanced_metadata_file: Path,
    model_id: str,
    model_name: str,
    cache_directory: Path,
    start_idx: int,
    end_idx: int,
    hf_token: str | None = None,
) -> None:
    output_enhanced_dir.mkdir(parents=True, exist_ok=True)
    enhanced_cases_dir.mkdir(parents=True, exist_ok=True)

    with en_metadata_file.open(encoding="utf-8") as f:
        cases = json.load(f)
    print(f"📖 Loaded {len(cases)} case records")

    todo = []
    for rec in cases[start_idx:end_idx]:
        case_id = rec["case_id"]
        out_path = enhanced_cases_dir / f"{case_id}_{model_name}.json"
        if not out_path.exists():
            todo.append(rec)
    if not todo:
        print(f"✅ All cases in range {start_idx}-{end_idx - 1} already processed. Nothing to do.")
        return

    if hf_token:
        if login is None:
            raise ImportError("huggingface_hub is required for token login.")
        login(token=hf_token)

    if torch is None:
        raise ImportError("torch is required for case enhancement processing.")

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.reset()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    model, processor = load_model(model_id, cache_directory)

    print("🔍 After model load, GPU memory summary:", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
        free, total = torch.cuda.mem_get_info(i)
        print(f"→ Free: {free // (1024**2)} MiB / Total: {total // (1024**2)} MiB", flush=True)
        print(torch.cuda.memory_summary(device=i), flush=True)

    print("\n📍 Device map used by HuggingFace model:")
    try:
        print(model.hf_device_map)
    except Exception as e:
        print(f"⚠️ Could not print device map: {e}")

    enhanced_file = enhanced_metadata_file.with_name(
        enhanced_metadata_file.stem + f"_start_at_index_{start_idx}" + enhanced_metadata_file.suffix
    )
    enriched_cases = {}
    if enhanced_file.exists():
        print(f"🔁 Resuming from existing file → {enhanced_file.name}")
        if enhanced_file.stat().st_size > 0:
            try:
                with enhanced_file.open(encoding="utf-8") as f:
                    enriched_cases = {rec["case_id"]: rec for rec in json.load(f)}
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to load JSON: {e} — starting fresh.")
        else:
            print("⚠️ File exists but is empty — starting fresh.")
    else:
        print("🆕 No previous file found — starting fresh.")

    print(f"🧠 Previously processed: {len(enriched_cases)} cases")
    print(enhanced_file)

    for _, rec in enumerate(
        tqdm(
            cases[start_idx:end_idx],
            desc=f"Cases {start_idx}-{end_idx - 1}",
            total=max(0, min(end_idx, len(cases)) - start_idx),
        )
        if tqdm is not None
        else cases[start_idx:end_idx]
    ):
        case_id = rec["case_id"]
        out_path = enhanced_cases_dir / f"{case_id}_{model_name}.json"
        if out_path.exists():
            continue

        txt = (rec.get("full_text_en") or rec.get("full_text") or "").replace("\n", " ")
        if not txt.strip():
            continue

        try:
            rec["genAI_legal_summary"] = process_case(
                text=txt,
                case_id=case_id,
                model=model,
                processor=processor,
                output_enhanced_dir=output_enhanced_dir,
            )
            enriched_cases[case_id] = rec

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)

            with enhanced_file.open("w", encoding="utf-8") as f:
                json.dump(list(enriched_cases.values()), f, indent=2, ensure_ascii=False)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  {case_id} failed: {e}")
