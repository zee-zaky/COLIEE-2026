#!/bin/bash
set -euo pipefail
echo "🔄  START: full clean-rebuild of Conda env 'coliee'"

ENV_NAME=coliee
NOBACKUP=/nesi/nobackup/uoa04665/mzak071

echo "🗑️   Deleting old env & caches…"
rm -rf ~/.conda/envs/$ENV_NAME
rm -rf "$NOBACKUP/conda_pkgs" "$NOBACKUP/tmp" "$NOBACKUP/hf_home"
mkdir -p "$NOBACKUP/conda_pkgs" "$NOBACKUP/tmp" "$NOBACKUP/hf_home"

echo "📁  Setting cache locations under $NOBACKUP"
export TMPDIR="$NOBACKUP/tmp"
export CONDA_PKGS_DIRS="$NOBACKUP/conda_pkgs"
export HF_HOME="$NOBACKUP/hf_home"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

echo "📦  Loading Miniforge3 (recommended by NeSI)…"
module --force purge
module load Miniforge3/25.3.1-0   # adjust if needed
source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1

echo "⚙️   Configuring conda pkgs_dirs → $CONDA_PKGS_DIRS"
conda config --set env_prompt '({name})'
conda config --remove-key pkgs_dirs 2>/dev/null || true
conda config --add pkgs_dirs "$CONDA_PKGS_DIRS"

YAML="$TMPDIR/env.yml"
echo "📝  Writing YAML spec → $YAML"
cat > "$YAML" <<EOF
name: $ENV_NAME
channels:
  - conda-forge
dependencies:
  - python=3.10
  - transformers>=4.40
  - accelerate
  - huggingface_hub
  - safetensors
EOF

echo "🚀  Creating env…"
# Prefer mamba if present; otherwise conda works (slower).
if command -v mamba >/dev/null 2>&1; then
  mamba env create -f "$YAML"
else
  conda env create -f "$YAML"
fi

echo "🔍  Activating env & printing versions…"
conda activate "$ENV_NAME"

python - <<'PY'
import transformers, os, sys, platform
print("✅  Env OK")
print("   Python        :", platform.python_version())
print("   Transformers  :", transformers.__version__)
print("   HF cache      :", os.environ.get("TRANSFORMERS_CACHE"))
print("   Env location  :", sys.prefix)
PY

echo "🎉  DONE: Conda environment '$ENV_NAME' rebuilt."
