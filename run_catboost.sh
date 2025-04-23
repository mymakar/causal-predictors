#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=cataboost.put
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --account=precisionhealth_owned1
#SBATCH --partition=precisionhealth
#SBATCH --mem-per-gpu=500gb


python scripts/train_catboost_optuna_headroom.py \
    --experiment college_scorecard \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --results_dir /nfs/turbo/coe-rbg/mmakar/causalfeatures/results \
    --use_cached \
    --use_gpu \
    --split_mode $1 