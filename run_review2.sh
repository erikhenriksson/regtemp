#!/bin/bash
#SBATCH --account=project_462001491
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/review2_%j.out
#
# Analyses for the second review, after runs/test_400m_meta_only_rich
# (metadata-only model, dev and test) and runs/logreg_probs (logistic
# regression with probabilities) exist. Run from the repo:
#
#     sbatch run_review2.sh
#
# 1. Register shares by year on the held-out documents and on the test
#    documents alone, dated by htmldate, the crawl year, the final model,
#    the metadata-only model and the logistic regression; compared with
#    htmldate over 2011-2025 and 2000-2025.
# 2. Test diagnostics with the whole predicted distribution.
# 3. How often htmldate finds a date, by register and crawl period.

source ./setup_lumi.sh
set -e
P=runs/predict_400m_meta_start_all
RUNS="--run meta_only=runs/test_400m_meta_only_rich --run logreg=runs/logreg_probs"

for DOCS in heldout test; do
    python 10_register_trends.py --dated $DATA/hplt.dated.jsonl --predictions $P \
        --docs $DOCS --periods 1995..2025 $RUNS --out runs/years_${DOCS}_models.csv
    for FROM in 2011 2000; do
        python compare_trends.py runs/years_${DOCS}_models.csv --from-year $FROM
    done
done

python 13_test_diagnostics.py --predictions $P --dated $DATA/hplt.dated.jsonl \
    --results runs/test_diagnostics
python 16_dateability.py --dated $DATA/hplt.dated.jsonl --out runs/dateability.csv
