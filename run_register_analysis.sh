#!/bin/bash
#SBATCH --account=project_462001491
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/register_analysis_%j.out
#
# All register analyses for the paper, after 9_predict_years.py has dated the
# sample. Run from the repo:
#
#     sbatch run_register_analysis.sh
#
# Shares by period (tables) and by year (figures), with htmldate, crawl and
# classifier dates; the paper figures; the comparisons with htmldate; the
# test-set diagnostics; the age of the sample; the annotation sample.

source ./setup_lumi.sh
set -e
P=runs/predict_400m_meta_start_all
D=$DATA/hplt.dated.jsonl

# shares by period
python 10_register_trends.py --dated $D --docs dated --out runs/registers_dated.csv
python 10_register_trends.py --dated $D --predictions $P --docs heldout --out runs/registers_heldout.csv
python 10_register_trends.py --predictions $P --docs unlabelled --out runs/registers_unlabelled.csv

# shares by year
python 10_register_trends.py --dated $D --docs dated --periods 1995..2025 --out runs/years_dated.csv
python 10_register_trends.py --dated $D --docs dated --level sub --periods 1995..2025 --out runs/years_dated_sub.csv
python 10_register_trends.py --dated $D --predictions $P --docs heldout --periods 1995..2025 --out runs/years_heldout.csv
python 10_register_trends.py --predictions $P --docs unlabelled --periods 1995..2025 --out runs/years_unlabelled.csv

# paper figures (paper.pdf in each folder)
python 11_plot_register_trends.py --line htmldate=runs/years_dated.csv:htmldate \
    --sub runs/years_dated_sub.csv:htmldate --registers NA,IN,IP,ID --from-year 2000 --out runs/paper_dated
python 11_plot_register_trends.py --line htmldate=runs/years_heldout.csv:htmldate \
    --line classifier=runs/years_heldout.csv:model_prob --line "most likely year=runs/years_heldout.csv:model" \
    --registers NA,IN,IP,ID --from-year 2000 --out runs/paper_heldout
python 11_plot_register_trends.py --line "dated, htmldate=runs/years_dated.csv:htmldate" \
    --line "rest of sample, classifier=runs/years_unlabelled.csv:model_prob" \
    --registers NA,IN,IP,ID --from-year 2000 --out runs/paper_population
python 11_plot_register_trends.py --line htmldate=runs/years_dated.csv:htmldate \
    --line "crawl date=runs/years_dated.csv:crawl" \
    --registers NA,IN,IP,ID --from-year 2000 --out runs/paper_crawl

# comparisons with htmldate, by year
python compare_trends.py runs/years_heldout.csv --from-year 2000
python compare_trends.py runs/years_heldout.csv --from-year 2011
python compare_trends.py runs/years_dated.csv --from-year 2000

# test-set diagnostics, age of the sample, annotation sample
python 13_test_diagnostics.py --predictions $P --dated $D --results runs/test_diagnostics
python 12_age_of_sample.py --predictions $P --dated $D --results runs/age
python 14_annotation_sample.py --dated $D --predictions $P --out runs/annotation
