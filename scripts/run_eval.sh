#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=eval_%j.log
#SBATCH --error=eval_%j.log

# Account MaxWall is 2h -> the eval set is split into shards, one job each.
# Submit all shards:
#   for i in 0 1 2 3; do sbatch run_eval.sh mcdc  mcdc_epoch36.pt $i 4; done
#   for i in 0 1 2 3; do sbatch run_eval.sh bayes bayes_best.pt  $i 4; done
# Then merge + plots (login node):
#   python3 analyze.py --csv "mcdc=results/mcdc_shard*_eval.csv" ...

MODEL=${1:-mcdc}
CKPT=${2:-mcdc_epoch36.pt}
SHARD=${3:-0}
NSHARDS=${4:-4}
TAG=${5:-$MODEL}      # output prefix in results/ — set to avoid clobbering other runs of the same model type

cd "$SLURM_SUBMIT_DIR"
source ../newenv/bin/activate
module load cuda

python evaluate.py \
    --model  "$MODEL" \
    --ckpt   "$CKPT" \
    --mc_samples 30 \
    --num_workers 8 \
    --shard_idx "$SHARD" \
    --num_shards "$NSHARDS" \
    --tag "$TAG" \
    --out_dir results
