#!/bin/bash
set -euo pipefail
echo "🔄  START: full clean-rebuild of Conda env 'coliee'"

##############################
# 0. Purge old stuff
##############################
ENV_NAME=coliee
NOBACKUP=/nesi/nobackup/uoa04665/mzak071

echo "🗑️   Deleting old env & caches…"
rm -rf ~/.conda/envs/$ENV_NAME
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

# Try to discover an available Miniforge3 module and load it.
# This avoids hardcoding a version that isn't on this cluster/software stack.
MF_LINE="$(module -t avail Miniforge3 2>&1 | grep -E '^Miniforge3/' | head -n 1 || true)"

if [[ -z "$MF_LINE" ]]; then
  echo "❌  No Miniforge3 module found via 'module avail Miniforge3'."
  echo "    Run: module spider Miniforge3"
  exit 1
fi

echo "✅  Loading: $MF_LINE"
module load "$MF_LINE"

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1


echo "⚙️   Configuring conda pkgs_dirs → $CONDA_PKGS_DIRS"
conda config --set env_prompt '({name})'
conda config --remove-key pkgs_dirs 2>/dev/null || true
conda config --add pkgs_dirs "$CONDA_PKGS_DIRS"

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
