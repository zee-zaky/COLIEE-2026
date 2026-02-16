#!/bin/bash
# slurm/setup_coliee_conda_env.sh   — clean rebuild with progress echoes

set -euo pipefail
echo "🔄  START: full clean-rebuild of Conda env 'coliee'"

##############################
# 0. Purge old stuff
##############################
ENV_NAME=coliee
NOBACKUP=/nesi/nobackup/uoa04341/mzak071

#echo "🗑️   Deleting old env & caches…"
#rm -rf ~/.conda/envs/$ENV_NAME
#rm -rf $NOBACKUP/{conda_pkgs,tmp,hf_cache}
#mkdir -p $NOBACKUP/{conda_pkgs,tmp,hf_cache}

##############################
# 1. Set cache variables
##############################
echo "📁  Setting cache locations under $NOBACKUP"
export TMPDIR="$NOBACKUP/tmp"
export CONDA_PKGS_DIRS="$NOBACKUP/conda_pkgs"
export TRANSFORMERS_CACHE="$NOBACKUP/hf_cache"

##############################
# 2. Load Conda/Mamba modules
##############################
echo "📦  Loading Miniconda & Mamba modules…"
module purge
module load Miniconda3/23.10.0-1
module load Mamba/23.1.0-1

echo "⚙️   Configuring conda pkgs_dirs → $CONDA_PKGS_DIRS"
conda config --set env_prompt '({name})'
conda config --add pkgs_dirs "$CONDA_PKGS_DIRS"

##############################
# 3. Create minimal env
##############################
YAML=$TMPDIR/env.yml
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

echo "🚀  mamba env create (this can take a minute)…"
mamba env create -f "$YAML"

##############################
# 4. Quick sanity check
##############################
echo "🔍  Activating env & printing versions…"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python - <<'PY'
import transformers, os, sys, platform
print("✅  Env OK — CPU-only build")
print("   Python        :", platform.python_version())
print("   Transformers  :", transformers.__version__)
print("   HF cache      :", os.environ["TRANSFORMERS_CACHE"])
print("   Env location  :", sys.prefix)
PY

echo "🎉  DONE: Conda environment '$ENV_NAME' rebuilt."
echo "   * Environment folder : ~/.conda/envs/$ENV_NAME"
echo "   * Conda pkgs cache   : $CONDA_PKGS_DIRS"
echo "   * HF model cache     : $TRANSFORMERS_CACHE"
echo "   * Temp build dir     : $TMPDIR"

