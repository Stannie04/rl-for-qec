#!/bin/bash

#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --job-name=qec-train
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --output=/home/s3208761/slurm/cache/%x_%j.out

module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1

source activate /home/s3208761/conda_envs/qugym

AGENT="dqn"
CODE="18_2_3_toric"


#pip freeze
python -c "import torch; print('GPU available?', torch.cuda.is_available())"
python main.py $AGENT $CODE
