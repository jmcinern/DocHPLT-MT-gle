#!/bin/bash
#SBATCH --job-name=hplt_download
#SBATCH --account=project_465002364
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/project_465002364/DocHPLT-MT-gle/logs/hplt_download_%j.out
#SBATCH --error=/scratch/project_465002364/DocHPLT-MT-gle/logs/hplt_download_%j.err

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
REPO=/scratch/project_465002364/DocHPLT-MT-gle
export HF_HOME=/scratch/project_465002364/hf_cache

mkdir -p $REPO/logs $REPO/data/train $HF_HOME

singularity exec \
    -B /scratch/project_465002364:/scratch/project_465002364 \
    $SIF \
    python $REPO/scripts/download_training_data.py \
    --output_dir $REPO/data/train

echo "Job finished: $(date)"
