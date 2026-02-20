
import sys          
import argparse
import os
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import re
import unicodedata
from typing import List, Dict,  Tuple, Union, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login
import gc
from tqdm import tqdm
import asyncio
from accelerate import infer_auto_device_map
import subprocess
#from ollama import Client
import pprint
import time
from huggingface_hub import snapshot_download
import openai
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login
from PIL import Image
import requests
import torch
import transformers
import torch, os, socket
torch.set_float32_matmul_precision('high')
import ast
import logging
from pathlib import Path
import gc
from config.paths import PRESETS



parser = argparse.ArgumentParser(
    description="Process a slice of cases for legal-metadata extraction"
)
parser.add_argument(
    "--preset",
    choices=sorted(PRESETS.keys()),
    default="test_2026",
    help="Preset dataset configuration."
)
parser.add_argument("--start", type=int, default=0,
                    help="0-based index of the first case to process")
parser.add_argument("--num", type=int, default=10,
                    help="How many cases to handle in this run")
parser.add_argument("--offsit", type=int, default=0,
                    help="Offset to skip N cases before starting")
parser.add_argument(
    "--hf-token",
    type=str,
    default=None,
    help="Hugging Face token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN)."
)


args = parser.parse_args()
cfg = PRESETS[args.preset]

# Redirect Hugging Face hub cache using configured preset path
os.environ["HF_HOME"] = str(cfg.hf_cache_dir)
os.environ["HF_HUB_CACHE"] = str(cfg.hf_cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cfg.hf_cache_dir)

# 1. Print current GPU status
print("🖥️  GPU status via nvidia-smi:")
try:
    subprocess.run(["nvidia-smi"], check=True)
except subprocess.CalledProcessError as e:
    print(f"⚠️ nvidia-smi failed: {e}")



for i in range(torch.cuda.device_count()):
    mem = torch.cuda.mem_get_info(i)
    print(f"GPU {i}: free {mem[0] // (1024**2)} MiB / total {mem[1] // (1024**2)} MiB", flush=True)
    print(torch.cuda.memory_summary(device=i), flush=True)




#──────────────────────────────────
# Settings, Directories and Files
#──────────────────────────────────
CASES_DIR = cfg.cases_dir
LABELS_FILE = cfg.labels_file
OUTPUT_BASE_DIR = cfg.output_clean_dir
OUTPUT_ENHANCED_DIR = cfg.output_enhanced_dir

OUTPUT_CASES_DIR = cfg.cleaned_cases_dir
METADATA_FILE = cfg.metadata_file
EN_METADATA_FILE = cfg.metadata_en_file
MA_METADATA_DIR = cfg.ma_metadata_dir
OUTPUT_LOGS_DIR = cfg.enhanced_logs_dir
OUTPUT_ENHANCED_DIR.mkdir(exist_ok=True)
OUTPUT_BASE_DIR.mkdir(exist_ok=True)
OUTPUT_CASES_DIR.mkdir(parents=True, exist_ok=True)
CASES_OUTPUT_DIR = cfg.enhanced_cases_dir
CASES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

model_id = "google/gemma-3-27b-it"
MODEL_NAME = "gemma-3-27b-it"
MAX_CONTEXT = 128_000


ENHANCED_METADATA_FILE = OUTPUT_ENHANCED_DIR / f"{cfg.enhanced_metadata_prefix}_{MODEL_NAME.replace(':','_')}.json"
# Use configured cache location
cache_directory = str(cfg.hf_cache_dir)


# Skip first 3000 cases
start_idx  = args.offsit + args.start
end_idx    = start_idx + args.num          # non-inclusive


#────────────────────────────────────────
# Load all English-processed cases 
#────────────────────────────────────────

with EN_METADATA_FILE.open(encoding="utf-8") as f:
    cases = json.load(f)

print(f"📖 Loaded {len(cases)} case records")

#────────────────────────────────────────────────────────────────────
# Early-exit check: are all target cases already processed?
#────────────────────────────────────────────────────────────────────
todo = []
for rec in cases[start_idx:end_idx]:
    case_id   = rec["case_id"]
    out_path  = CASES_OUTPUT_DIR / f"{case_id}_{MODEL_NAME}.json"
    if not out_path.exists():          # needs work
        todo.append(rec)

if not todo:
    print("✅ All cases in range "
          f"{start_idx}-{end_idx-1} already processed. Nothing to do.")
    sys.exit(0)         # skip model load & the rest of the script





# Log in to Hugging Face (only needed once per session)
hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    raise ValueError(
        "Missing Hugging Face token. Pass --hf-token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
    )
login(token=hf_token)


#Increase TorchDynamo's cache limit to handle more recompilations
torch._dynamo.config.cache_size_limit = 128  # Increase from default 64
torch._dynamo.reset()  # Clear existing cache
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # memory settings FIRST
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error reporting
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Disable TorchDynamo

# Load model using specified cache directory
def load_model():

    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total_mem = torch.cuda.get_device_properties(i).total_memory
        max_memory[i] = int(total_mem * 0.8)  # Use only 60% of VRAM
    
    max_memory["cpu"] = 200 * 1024 ** 3  # 100GB in bytes
    print(max_memory)

    
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_directory,
            device_map="auto",
            max_memory=max_memory,  # Critical for preventing OOM
            #max_memory={0: "70GiB", "cpu": "300GiB"},
            torch_dtype=torch.bfloat16,  # Use FP32 for stability
            trust_remote_code=True,
            top_k=None,
            top_p=None,
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        return model, processor        
    except RuntimeError as e:
        print(f"❌ Model loading failed: {e}")
        raise

    
model, processor = load_model()



print("🔍 After model load, GPU memory summary:", flush=True)
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
    free, total = torch.cuda.mem_get_info(i)
    print(f"→ Free: {free // (1024**2)} MiB / Total: {total // (1024**2)} MiB", flush=True)
    print(torch.cuda.memory_summary(device=i), flush=True)


# 2. Print device map used by HuggingFace model
print("\n📍 Device map used by HuggingFace model:")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_info()
    print(model.hf_device_map)
except Exception as e:
    print(f"⚠️ Could not print device map: {e}")


#──────────────────────────────────
# Each entry defines: key, return format, short instruction
#──────────────────────────────────
ASPECT_SPECS = [
    (
    "judge",
    '["list of judge name(s) exactly as written in the case text; empty list if not found"]',
    "Extract the name(s) of the presiding judge(s) or decision-maker(s) if explicitly stated in the case text. "
    "Do NOT return names of judges mentioned only in cited/previous cases. "    
    "Return the judge name(s) exactly as written in the case text who issued the decision on this case not the previous cited ones. "
    "If no judge name is present, return an empty list."
    ),
    (
        "statutes", 
        "[list of extracted statutes from the case text (including mentioned year and sections in one item together)]", 
        #"[list of extracted statutes from the case text (strcuture it to include name, year and list of mentioned sections)]", 
        #"[{ 'name': <full statute name>, 'year': <year>, 'sections': [<section refernced in the case text>] }]",         
        "Extract DISTINCT statutes (laws) or subordinate instrument  (containing the word act / regulations / rules ... etc) that is **explicitly named**  and mentioned in the case text"
        # """Extract DISTINCT statutes, acts, or regulations explicitly mentioned in the case text. 
        #     These include anything named with 'Act', 'Regulations', or 'Rules'. 
        #     Where possible, extract:
        #     - Full name (e.g., 'Immigration and Refugee Protection Act')
        #     - Year (if mentioned)
        #     - Sections (if specific ones are cited)
        #     Return the output as a list of structured JSON objects with keys: name, year, sections."""
    ),
    (
        "legal_issues", 
        "['list of legal issues addressed in the case, stated in an abstracted and generalised manner']",
        "Identify and list the core legal issues addressed in the case. Phrase each issue abstractly, without using party names or case-specific details. Focus on the legal questions the court needed to resolve, suitable for metadata extraction."
    ),
    # (
    #     "citations",
    #     "['list of other cases citations mentioned in the text, don't include masked citations FRAGMENT_SUPPRESSED or REFERENCE_SUPPRESSED']",
    #     "Extract all case citations explicitly mentioned in the case text. Include full legal citations such as decisions from the Supreme Court, Federal Court, tribunals, or any reported/unreported case references. Do not include statutes or legislation—only prior case citations. Preserve their formatting as-is where possible."
    # ),
    (
        "legal_topics", 
        "['list of broad doctrinal labels, not case-specific names']", 
        "Identify and list the overarching legal doctrines or thematic areas addressed in the case. Use broad, abstract terms. Avoid using party names, statute names, or case-specific phrasing. Focus on categorising the legal reasoning into general doctrinal areas suitable for metadata extraction."
    ),
    (
        "factual_background", 
        "['list of items summarising the factual background of the case in an abstract, generalised manner']", 
        "Extract only the factual background from the case text. Summarise the relevant events, actors, and circumstances that gave rise to the legal dispute. Avoid legal analysis, arguments, or conclusions. Write in an abstracted, case-neutral tone suitable for metadata extraction."
    ),
    (
        "key_arguments", 
        '''{{
                "Applicants": ["list of items covering all arguments presented by the applicant(s)"],
                "Respondents": ["list of items covering all arguments presented by the respondent(s)"]
            }}''', 
        "Extract and summarise the key legal and factual arguments made by each side in the case. Separate the arguments clearly by party. Avoid paraphrasing the court’s evaluation or findings—focus strictly on what each party claimed or contested."
    ),
    (
        "court_analysis", 
        "['list of items explaining what the court found and *why and how* it reached its decision']", 
        "Summarise the court’s reasoning and analysis. Focus on the key findings and the legal or factual rationale behind the decision. Explain how the court interpreted statutes, weighed evidence, or applied legal principles to resolve the issues. Avoid repeating party arguments—focus on the court’s own logic and conclusions."
    ),
    (
        "final_outcome", 
        "['list of outcomes or orders issued by the court']", 
        "Identify and list the final outcome(s) or orders made by the court. Focus on the court’s binding decisions—e.g., whether the application was granted or dismissed, any declaratory or injunctive relief, cost orders, or directions to parties. Use neutral and formal phrasing. Do not include reasoning or background—only the court's final determinations suitable for metadata extraction."
    ),
]




def build_prompt(aspect_key, format_snippet, instruction):
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
PROMPTS = {
    aspect_key: build_prompt(aspect_key, format_snippet, instruction)
    for (aspect_key, format_snippet, instruction) in ASPECT_SPECS
}

print(PROMPTS["statutes"])



# ── Group definitions ──────────────────────────────────────────
# 1. Build lookup of aspect_key → instruction
ASPECT_INSTRUCTIONS = {
    aspect_key: instruction
    for (aspect_key, _, instruction) in ASPECT_SPECS
}

# 2. Define groups using just keys
# GROUP_KEYS = {
#     "group1": ["statutes", "legal_issues", "citations","legal_topics"],
#     "group2": ["factual_background", "key_arguments"],
#     "group3": ["court_analysis", "final_outcome"],
# }

GROUP_KEYS = {
     "group1": [
         "judge",
         "statutes", 
         #"citations",
         "legal_issues", 
         "legal_topics"
         ],
     "group2": [
         "factual_background", 
         "key_arguments", 
         "court_analysis", 
         "final_outcome"
         ]     
 }

ORDERED_KEYS = [    
    "judge",
    "statutes",
    "legal_issues",
    #"citations",
    "factual_background",
    "key_arguments",
    "court_analysis",
    "final_outcome",
    "legal_topics",
]


# 3. Build GROUP_SPECS dynamically using the reused instructions
GROUP_SPECS = []
for group_name, aspect_keys in GROUP_KEYS.items():
    instructions = "\n".join(f"- {ASPECT_INSTRUCTIONS[k]}" for k in aspect_keys)
    group_instruction = f"You are a legal metadata expert. Extract the following:\n{instructions}"
    GROUP_SPECS.append((group_name, aspect_keys, group_instruction))

ASPECT_FORMATS = {
    aspect_key: format_snippet
    for (aspect_key, format_snippet, _) in ASPECT_SPECS
}

def build_group_prompt(keys, instruction):    
    field_block = ",\n  ".join(f'"{k}": {ASPECT_FORMATS[k]}' for k in keys)
    return f"""{instruction}

Return a JSON object with:
{{{{
  {field_block}
}}}}
(no extra keys or commentary)

=======================
CASE TEXT STARTS
=======================
{{TEXT}}
=======================
CASE TEXT ENDS
=======================
Remember, as legal metadata expert you need to extract the requested information in the instructions and return the information in a structured JSON object with the following format:
{{{{
  {field_block}
}}}}
(no extra keys or commentary)
"""

# dict keyed by group name → prompt
GROUP_PROMPTS = {
    gname: build_group_prompt(keys, instr)
    for gname, keys, instr in GROUP_SPECS
}

print(GROUP_PROMPTS["group1"])

#────────────────────────────────────────
# Helper: send prompt, parse JSON safely
#────────────────────────────────────────


def setup_logger(case_id: str, log_dir: Path):
    """Configure a dedicated logger for a single case."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case_id}.log"

    # Remove any existing handlers to avoid duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True  # overrides any previous config
    )


def query_aspect(aspect: str, tmpl: str, text: str) -> dict:
    """Send a single-aspect prompt to Gemma and return structured JSON."""
    
    prompt = tmpl.format(TEXT=text)

    # Log the prompt being sent
    logging.info("#────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────#")
    logging.info(f"\n\n🟦 Aspect: {aspect}\n--- Prompt ---\n{prompt}")

    # 1. Format messages in Gemma expected format
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a legal metadata expert."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    # 2. Prepare inputs with processor
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        #padding=MAX_CONTEXT,
        #max_length=MAX_CONTEXT,
        #truncation=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    # 3. Generate output
    with torch.inference_mode():
        output = model.generate(
            **inputs, 
            max_new_tokens=4096, 
            do_sample=False,
            top_p=None,  # explicitly disable if injected
            top_k=None
        )
        output_ids = output[0][input_len:]

    # 4. Decode
    raw = processor.decode(output_ids, skip_special_tokens=True)

    logging.info(f"--- Raw output for `{aspect}` ---\n{raw}")

    # 5. Extract JSON from output (same logic as before)
    first, last = raw.find("{"), raw.rfind("}")
    if first == -1 or last == -1:
        error_msg = f"Couldn’t find JSON in output for `{aspect}`:\n{raw}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    json_str = raw[first : last + 1]

    try:
        parsed = json.loads(json_str)
        logging.info(f"--- Parsed JSON for `{aspect}` ---\n{json.dumps(parsed, indent=2)}")
        return parsed
    except json.JSONDecodeError as e:
        logging.warning(f"Standard JSON parsing failed: {e}\nTrying ast.literal_eval fallback...")
        try:
            # Use ast.literal_eval as a fallback (safe for dicts/lists with single quotes)
            parsed = ast.literal_eval(json_str)
            logging.info(f"--- Parsed with ast.literal_eval for `{aspect}` ---\n{parsed}")
            return parsed
        except Exception as ast_e:
            error_msg = f"❌ Fallback parsing also failed for `{aspect}`: {ast_e}\n\nRaw:\n{raw}"
            logging.error(error_msg)
            raise ValueError(error_msg)



def process_case(text: str, case_id: str) -> dict:
    """Run all aspect prompts, merge the fragments."""
    # Setup case-specific log
    log_dir = OUTPUT_LOGS_DIR
    setup_logger(case_id, log_dir)
    logging.info(f"\n\n🗂️ Processing case ID: {case_id}")

    
    aggregated = {}
    for aspect, tmpl in PROMPTS.items():
        try:
            piece = query_aspect(aspect, tmpl, text)
            if aspect == "key_arguments":
                aggregated["key_arguments"] = piece["key_arguments"]
            else:
                aggregated[aspect] = piece[aspect]
        except Exception as e:
            error_msg = f"⚠️ Skipped `{aspect}` in case `{case_id}` due to: {e}"
            logging.error(error_msg)
            raise ValueError(error_msg)
    return aggregated


# ────────────────────────────────────────
# Load previous progress if available
# ────────────────────────────────────────



# Inject start index into filename
ENHANCED_METADATA_FILE = ENHANCED_METADATA_FILE.with_name(
    ENHANCED_METADATA_FILE.stem + f"_start_at_index_{start_idx}" + ENHANCED_METADATA_FILE.suffix
)

enriched_cases = {}
if ENHANCED_METADATA_FILE.exists():
    print(f"🔁 Resuming from existing file → {ENHANCED_METADATA_FILE.name}")

    if ENHANCED_METADATA_FILE.stat().st_size > 0:
        try:
            with ENHANCED_METADATA_FILE.open(encoding="utf-8") as f:
                enriched_cases = {rec["case_id"]: rec for rec in json.load(f)}
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to load JSON: {e} — starting fresh.")
    else:
        print("⚠️ File exists but is empty — starting fresh.")
else:
    print("🆕 No previous file found — starting fresh.")

print(f"🧠 Previously processed: {len(enriched_cases)} cases")
print(ENHANCED_METADATA_FILE)


# ────────────────────────────────────────
# Start processing remaining cases
# ────────────────────────────────────────
start = time.time()
total = len(cases)

counter = 0



#for i, rec in enumerate(tqdm(cases[start_index:], desc="Summarising cases", total=total - start_index), start=start_index):

for i, rec in enumerate(
        tqdm(cases[start_idx:end_idx],
             desc=f"Cases {start_idx}-{end_idx-1}",
             total=min(args.num, len(cases) - start_idx))):


    case_id = rec["case_id"]

    out_path = CASES_OUTPUT_DIR / f"{case_id}_{MODEL_NAME}.json"
    # 🔁 Skip if already written to individual file
    if out_path.exists():
        continue

    txt = (rec.get("full_text_en") or rec.get("full_text") or "").replace("\n", " ")
    if not txt.strip():
        continue

    try:
        rec["genAI_legal_summary"] = process_case(txt, case_id)
        enriched_cases[case_id] = rec

        # Save individual case file
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  {case_id} failed: {e}")
