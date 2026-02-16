#!/bin/bash
set -euo pipefail
echo "🔄  START: full clean-rebuild of Conda env 'coliee'"

##############################
# 0. Purge old stuff
##############################
ENV_NAME=coliee
NOBACKUP=/nesi/nobackup/uoa04665/mzak071

echo "🗑️   Deleting old env & caches…"
rm -rf "$NOBACKUP/conda_envs/$ENV_NAME"
rm -rf "$NOBACKUP/conda_pkgs" "$NOBACKUP/tmp" "$NOBACKUP/hf_home"
mkdir -p "$NOBACKUP/conda_pkgs" "$NOBACKUP/tmp" "$NOBACKUP/hf_home"

##############################
# 1. Set cache variables
##############################
echo "📁  Setting cache locations under $NOBACKUP"
export TMPDIR="$NOBACKUP/tmp"
export CONDA_PKGS_DIRS="$NOBACKUP/conda_pkgs"
export HF_HOME="$NOBACKUP/hf_home"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

##############################
# 2. Load Conda/Mamba modules
##############################
echo "📦  Loading Miniforge3 (recommended by NeSI)…"
module --force purge

# Often required on NeSI; harmless if not needed on your node:
module load NeSI/zen3 2>/dev/null || true

# Load newest available Miniforge3 explicitly
module load Miniforge3/25.3.1-0

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1

# ---- Keep ALL conda state off $HOME ----
export CONDARC="$NOBACKUP/condarc"
export CONDA_ENVS_PATH="$NOBACKUP/conda_envs"
mkdir -p "$CONDA_ENVS_PATH"


# Write a minimal condarc in nobackup (idempotent)
cat > "$CONDARC" <<EOF
env_prompt: '({name})'
pkgs_dirs:
  - $CONDA_PKGS_DIRS
channels:
  - conda-forge
channel_priority: strict
EOF

echo "⚙️   Using CONDARC=$CONDARC"
echo "⚙️   Using CONDA_ENVS_PATH=$CONDA_ENVS_PATH"
echo "⚙️   Using CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"



##############################
# 3. Create minimal env
##############################
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

##############################
# 4. Quick sanity check
##############################
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
