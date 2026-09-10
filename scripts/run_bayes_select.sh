#!/bin/bash
#SBATCH --job-name=bayes_select
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bayes_select.log
#SBATCH --error=bayes_select.log

# Checkpoint selection for BayesNN: the training loop picks "best" from a
# SINGLE-sample val EER, which is noisy for a Bayesian net (large-sigma
# posteriors make individual samples bad even when the posterior mean of the
# predictive is good). Here we re-score the saved snapshots on the dev subset
# with a 10-sample MC estimate and print a table — pick the lowest MC EER for
# the full eval run.

# arg 1 = checkpoint prefix (default "bayes"), e.g.:  sbatch run_bayes_select.sh bayes_p01
PREFIX=${1:-bayes}

cd "$SLURM_SUBMIT_DIR"
source ../newenv/bin/activate
module load cuda

for ckpt in ${PREFIX}_epoch*.pt; do
    tag="select_$(basename "$ckpt" .pt)"
    echo "=== $ckpt ==="
    python evaluate.py \
        --model bayes \
        --ckpt "$ckpt" \
        --protocol_file LA/ASVspoof2019.LA.cm.dev.subset.txt \
        --flac_root     LA/ASVspoof2019_LA_dev/flac \
        --mc_samples 10 \
        --skip_det \
        --num_workers 8 \
        --tag "$tag" \
        --out_dir results/select
done

echo "=== selection table (dev subset, 10 MC samples) ==="
export PREFIX
python - <<'EOF'
import glob, json, os
rows = []
pattern = f"results/select/select_{os.environ['PREFIX']}_epoch*_summary.json"
for f in sorted(glob.glob(pattern)):
    s = json.load(open(f))
    rows.append((s["ckpt"], s["mc"]["eer"], s["mc"]["ece"], s["mc"]["nll"]))
rows.sort(key=lambda r: r[1])
print(f"{'ckpt':28s} {'EER%':>7s} {'ECE%':>7s} {'NLL':>7s}")
for c, eer, ece, nll in rows:
    print(f"{c:28s} {eer*100:7.2f} {ece*100:7.2f} {nll:7.3f}")
print(f"\nBEST: {rows[0][0]}")
EOF
