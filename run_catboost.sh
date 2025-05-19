#!/bin/bash
#SBATCH --time=200:00:00
#SBATCH --cpus-per-task=4

#SBATCH --output=catboost-%x.%j.out

#SBATCH --account=mmakar98
#SBATCH --partition=standard
#SBATCH --mem=100gb
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=rjsingh@umich.edu
#SBATCH --error=/nfs/turbo/coe-rbg/rjsingh/slurm/slurm-%A-%a.err

experiment=$1 
split=$2

errfile="/nfs/turbo/coe-rbg/rjsingh/slurm/catboost_d_${split}_${experiment}.err"
outfile="/nfs/turbo/coe-rbg/rjsingh/slurm_logs/catboost_d_${split}_${experiment}_${jobid}.out"

exec 2> "$errfile"
exec > "$outfile"

python scripts/train_catboost_optuna_headroom.py \
    --experiment $1 \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --results_dir /nfs/turbo/coe-rbg/rjsingh/results \
    --use_cached \
    --num_samples 20 \
    --split_mode $2 