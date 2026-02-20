#!/bin/bash -e

# ================================================================================================
# Example Command to submit the job array (adjust the array range and preset as needed):
# ================================================================================================
# sbatch \
#   --time=00:30:00 \
#   --array=0-1 \
#   --export=PRESET=test_2026,CASES_PER_TASK=1 \
#   slurm/job_array_run.sl \
#   --hf-token hf_xxx
#
# Alternative (token from environment instead of CLI argument):
# export HF_TOKEN=hf_xxx
# sbatch --array=0-49 --export=PRESET=test_2026,CASES_PER_TASK=20 slurm/job_array_run_20.sh
###############################################################################


#SBATCH --job-name=gemma_cases
#SBATCH --partition=milan
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=A100:1
#SBATCH --array=0-99
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --account uoa04665



###############################################################################
# Runtime parameters (passed via sbatch --export or environment variables)
###############################################################################

CASES_PER_TASK=${CASES_PER_TASK:-20}
PRESET=${PRESET:-test_2026}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "⚠️  This script expects to be run as an array job."
  echo "   Use: sbatch --array=0-49 ..."
  exit 1
fi

START_INDEX=$(( SLURM_ARRAY_TASK_ID * CASES_PER_TASK ))


echo "🧮 SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "📦 Processing cases $START_INDEX to $((START_INDEX + CASES_PER_TASK - 1))"
echo "📁 PRESET=${PRESET}"
echo "📊 CASES_PER_TASK=${CASES_PER_TASK}"

###############################################################################
# Optional HF token parameter
###############################################################################

HF_TOKEN_ARG=""
if [[ "${1:-}" == "--hf-token" ]]; then
  HF_TOKEN_ARG="${2:-}"
elif [[ -n "${1:-}" ]]; then
  HF_TOKEN_ARG="${1}"
fi

if [[ -n "${HF_TOKEN_ARG}" ]]; then
  echo "🔐 HF token provided via script parameter"
else
  echo "ℹ️ No HF token parameter provided; relying on env vars in runtime"
fi

###############################################################################
# Environment
###############################################################################

module load Miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /nesi/project/uoa04341/conda/nesi-env

###############################################################################
# Run
###############################################################################

CMD=(
  python 5.process_cases.py
  --preset "${PRESET}"
  --start "${START_INDEX}"
  --num "${CASES_PER_TASK}"
)

if [[ -n "${HF_TOKEN_ARG}" ]]; then
  CMD+=(--hf-token "${HF_TOKEN_ARG}")
fi

"${CMD[@]}"
