# RegTemp: dating HPLT documents and studying register change

Code for the paper *Text-Based Speed Dating: Scalable Dating for HPLT and
Register Change*. We label English HPLT 4.0 documents with publication dates
from `htmldate` on their original HTML, train a classifier that predicts the
publication year from what HPLT releases (text, URL, crawl date), and study
how web registers change over time.

## Pipeline

| Script | What it does |
|---|---|
| `1_fetch_html.py` | adds the original HTML to each HPLT document (Common Crawl byte ranges, Common Crawl's index files for wrong offsets, the Wayback Machine for Internet Archive crawls); resumable; `--direct-only` for the fast part first |
| `2_select_good.py` | keeps the first N documents whose HTML is correct (looks like HTML and contains HPLT's text), in sample order |
| `3_date_html.py` | htmldate on the HTML: `pub_date` (fast mode, the labels), `pub_date_extensive`, `crawl_date` |
| `4_date_without_html.py` | htmldate on URL, text, md and xml: how much of the date survives extraction |
| `5_summarize_dates.py` | tables for step 4, with crawl-year baselines |
| `6_train_classifier.py` | fine-tunes Ettin to predict the year; input options; `--trained` to score a saved model; `--evaluate-test` |
| `7_logreg_baseline.py` | TF-IDF + logistic regression baseline on the same split |
| `8_undated_estimate.py` | estimates the classifier's accuracy on documents htmldate could not date |
| `9_predict_years.py` | dates any HPLT file (e.g. all of `samples.jsonl`) with a trained classifier: most likely year, median, confidence, the full distribution over years; marks the htmldate-labelled lines and their split |
| `10_register_trends.py` | shares of the main registers by publication period, with htmldate years, classifier years (most likely, median, summed probabilities) or both on the same documents; bootstrap intervals |
| `0_time_fetch.py` | times retrieval as fast as the servers allow (cost of retrieving HTML) |
| `classifier_common.py` | shared by 6–10: loading, split, baselines, metrics |
| `score_all.sh` | test scores of every trained run, and the table of all runs |
| `cue_table.py`, `period_table.py` | the paper's tables by date cue and by period |
| `count_left.py`, `count_dates.py`, `check_html.py` | counts and checks used for the paper |
| `setup_lumi.sh` | environment for LUMI |

## How each result in the paper is produced

All commands run on LUMI in `/scratch/project_462001491/ehenriks/data/eng-1M`,
after `source setup_lumi.sh`. GPU steps need a LUMI-G node (one GCD); the
rest run on CPU. On LUMI some scripts were run under other names
(`6_train_classifier2.py`, `6_train_classifier3.py`, `6_train_classifier4.py`;
the final model `400m_meta_start_all` was trained with `6_train_classifier4.py`);
the repo's `6_train_classifier.py` is the latest version of the same script.

### Data files

| File | What it is | Made by |
|---|---|---|
| `samples.jsonl` | 1M English HPLT 4.0 documents, WDS 5–10, stratified by the sizes of the WDS buckets, shuffled. The first 150,000 lines are used. | (sampling, before this repo) |
| `hplt.html.jsonl` | the sample with HTML added (151,000 documents; the first 150,000 lines are complete) | `1_fetch_html.py samples.jsonl --out hplt.html.jsonl --n 150000` (first with `--direct-only`, then without) |
| `hplt.good.jsonl` | the 149,451 documents of the first 150,000 whose HTML is correct, in sample order | `2_select_good.py samples.jsonl hplt.html.jsonl --n 150000 --out hplt.good.jsonl` |
| `hplt.dated.jsonl` | the same with `pub_date` (fast mode, the labels), `pub_date_extensive` and `crawl_date` | `3_date_html.py hplt.good.jsonl --out hplt.dated.jsonl --workers 8` |
| `text_dates.jsonl` | htmldate run on URL, text, md, xml (dating without HTML) | `4_date_without_html.py hplt.dated.jsonl --out text_dates.jsonl --workers 8` |
| `runs/<name>/` | one trained classifier per folder (`best_model/`, `config.json`, dev metrics) | `6_train_classifier.py`, see below |
| `runs/test_<name>/` | test scores of each trained classifier | `score_all.sh` |
| `runs/predict_400m_meta_start_all/` | classifier dates for all 1M documents of `samples.jsonl` (`predictions.csv`, `probabilities.npy`, `years.json`) | `9_predict_years.py`, see "Register analysis" |

### Data and Methods

- **Sample (150,000 documents):** the first 150,000 lines of `samples.jsonl`.
- **Retrieving the HTML** — Common Crawl 133,292 (88.9%), Internet Archive
  16,708 (11.1%), index lookups 5,174 (3.4%), failed fetches 200:
  `python count_left.py samples.jsonl hplt.html.jsonl --n 150000`
  ("Input documents by crawl" and "Fetched, by how"). "Nearly all failures
  were Wayback pages" comes from the failed lines (198 of 200 were Wayback
  errors). Counts by route: commoncrawl 128,115, wayback 16,511,
  commoncrawl-index 5,174.
- **Retrieved 149,798, not retrieved 202, removed 347 (221 XML, 17 office
  documents and e-books, 14 other non-HTML, 95 mismatching pages), 149,451
  left:** the summary printed by `2_select_good.py` (command above). The 202
  are the 200 failed fetches plus 2 empty pages.
- **The HTML check** (five-word sequences) is the `verdict()` function in
  `2_select_good.py` (the same check as `check_html.py`).
- **Timing run (3,000 new documents):**
  `python 0_time_fetch.py samples.jsonl --start 500000 --n 3000 --out timing.jsonl --fetch-script 1_fetch_html.py --cache-dir hplt.html.jsonl.paths`.
  Printed: Common Crawl 2,661 documents in 8.1 min (5.49 docs/s), Wayback 339
  in 29.2 min (0.19 docs/s), 5 failed; projection for 10M documents 18.7 days
  (Common Crawl) and 67.6 days (Wayback). The 11.3% Internet Archive share
  is 339/3,000. The **52 days** at the full reported limit is computed:
  10M × 11.1% Internet Archive documents ÷ 15 requests per minute.
- **Pauses** (0.5 s Common Crawl, 4 s Wayback, index one at a time): settings
  in `0_time_fetch.py` and `1_fetch_html.py` (`BASE_PAUSE`).
- **htmldate speed (46 documents per second per core, 61 core-hours per 10M):**
  the progress line of `3_date_html.py` (365 docs/s with `--workers 8`, on a
  LUMI-C node, AMD EPYC 7763; fast and extensive mode together).
- **Classifier speed (about 135 documents per second, 21 GPU-hours per 10M):**
  `inference_docs_per_second` in `runs/test_400m_meta_start_all/config.json`
  (127.8 dev, 142.1 test; one MI250X GCD, 512 tokens, model only).
- **Split (83,285 / 10,307 / 10,342):** printed by `6_train_classifier.py`
  (hash of the document id, seed 42, `classifier_common.split_of`).

### Registers

Everywhere, a document's registers are those of
`classifier_common.register_labels`: every label of HPLT's `web-register`
field with probability at least 0.4, in the two-level taxonomy
(`classifier_common.TAXONOMY`). A subregister above the threshold is one
label with its main register ("HI re") and implies the main register; a
hybrid ("HI re" and "ID") counts in full towards each of its registers, so
register shares add up to more than 100%. `main_registers` gives the main
registers only ("ID HI"). Per-register results computed before this rule
(the most likely main register) are marked "to be redone" below.
`by_register.csv` of the runs in the table below still uses the old rule
until `score_all.sh` is run again (not used in the paper).

### Results

- **Dating HPLT with htmldate** — 103,934 of 149,451 dated (69.5%); 33.8% in
  the crawl year, 20.2% one year earlier, 35.9% three or more years earlier:
  the header and "lag" lines of the "All: all documents" table of
  `5_summarize_dates.py` (below); 35.9% = 3–5 years (17.6%) + 6 or more
  (18.3%). `count_dates.py hplt.dated.jsonl` gives the same counts.
- **How much of the date survives text extraction** (text and table
  `tab:text-dating`):
  `python 5_summarize_dates.py $DATA/text_dates.jsonl --by register --dated $DATA/hplt.dated.jsonl --metrics runs/metrics.csv --disagreements runs/check.csv`
  (from the repo; registers by threshold, see "Registers" below).
  Table columns: Found = `found`, Year P = `P yr`, Year R = `R yr`, from the
  "All: all documents" table (rows `url/fast`, `text/fast`, `text/extensive`,
  `text+url/fast`, `text+url/extensive`, `baseline/crawl year`). 26.3% = "year
  of the HTML date appears in the text". 53.6% = `jan 1` and 39.1% = `no ref`
  of `text/extensive`. 3.0% = `R yr` of `text+url/fast` in the "URL gives no
  date" table. Register sentences: "year appears in the text" line and
  `found` of `url/fast` in the per-register tables. **To be redone:** the
  numbers in the paper (NA 31.7%, HI 10.9%, IP 15.5%; NA 20.1%, OP 17.2%,
  IP 2.3%, ID 3.2%) used the most likely main register.
- **Main classifier table (`tab:classifier`) and appendix table of all runs
  (`tab:all-runs`):** `./score_all.sh` → `runs/all_results.csv` (dev and
  test exact, within one year, MAE, bias for every run). The rows "Always the
  crawl year" (34.5, 54.6, 2.78) and "Rule" (53.3, 68.7, 2.85) come from the
  "all" row of `cue_table.py` (below). The 1,024-token run was scored from its
  checkpoint (see the run list).
- **Text vs metadata, additions, model size, soft labels:** the same table.
- **Learning curve (56.5 / 58.1 / 59.5% on dev):** dev exact of `curve_20k`,
  `curve_40k` and `150m_meta_start_all` in `runs/all_results.csv`.
- **Bias (0.43 for 400M, 0.75 for 150M):** test bias in `runs/all_results.csv`.
- **Websites (25.8% seen; 57.5% seen, 62.3% unseen):**
  `grep -E "^test,(all|website).*,model," runs/test_400m_meta_start_all/metrics.csv`.
  "Even the crawl year is right less often for them": dev rows of
  `runs/150m_meta_startend/metrics.csv` (crawl year 25.4% for websites in
  train, 37.3% for the others).
- **Table by date cue (`tab:cues`):**
  `python cue_table.py runs/test_400m_meta_start_all/predictions_test.csv hplt.dated.jsonl --out runs/rule_vs_model.csv`.
- **Period table (`tab:periods`) and its 95% intervals:**
  `python period_table.py runs/test_400m_meta_start_all/predictions_test.csv --out runs/periods_test.csv`.
  The threshold 0.5 was chosen on the development set with the 150M model.
- **Documents that htmldate could not date** (cue shares 5.6% / 55.7%,
  register shares in `registers.csv`, confidence above 0.8: 13.0% vs 35.5%,
  estimates 54.6% and 48.5%):
  `python 8_undated_estimate.py $DATA/hplt.dated.jsonl --trained runs/400m_meta_start_all/best_model --results runs/undated_400m_meta_start_all`
  (from the repo, on a GPU node). **To be redone:** the register shares
  (31% vs 9% narrative), the reweighted estimate (54.6%) and the 57–59% below
  used the most likely main register; the cue shares and the confidence
  estimate do not depend on registers.
  The 23.1% of test documents with a URL year = 13.4% + 9.7% from
  `tab:cues`. The **57–59% across all documents** is computed: 69.5% of the
  documents are dated, so 0.695 × 61.0 + 0.305 × 48.5 = 57.2 and
  0.695 × 61.0 + 0.305 × 54.6 = 59.1.
- **Abstract:** 0.61 (final model, test), 0.53 (rule), 0.34 (crawl year).

### Register analysis

In progress; nothing from here is in the paper yet. From here on, scripts run
from the repo on LUMI (`/scratch/project_462001491/ehenriks/regtemp`) after
`source setup_lumi.sh`, which sets `$DATA` to the data folder. Models and
results are in the repo's `runs/` (not in git). The final model was copied
there from `$DATA/runs/400m_meta_start_all` (without its checkpoints).

- **Classifier dates for the 1M sample** (GPU, about 2 hours):

      sbatch --account=project_462001491 --partition=small-g --gpus-per-node=1 \
          --cpus-per-task=7 --mem=60G --time=04:00:00 --output=runs/predict_%j.out \
          --wrap "source ./setup_lumi.sh && python 9_predict_years.py \$DATA/samples.jsonl \
          --trained runs/400m_meta_start_all/best_model \
          --results runs/predict_400m_meta_start_all"

- **Register shares by period, htmldate years** (all dated documents):
  `python 10_register_trends.py --dated $DATA/hplt.dated.jsonl --docs dated --out runs/registers_dated.csv`
- **The same documents dated two ways** (dev and test: htmldate vs the
  classifier, the circularity check):
  `python 10_register_trends.py --dated $DATA/hplt.dated.jsonl --predictions runs/predict_400m_meta_start_all --docs heldout --out runs/registers_heldout.csv`
- **The whole sample with classifier dates** (without the 150,000 labelled
  lines; a different population from the dated documents):
  `python 10_register_trends.py --predictions runs/predict_400m_meta_start_all --docs unlabelled --out runs/registers_unlabelled.csv`

Registers: every label with probability at least 0.4 (`--threshold`); a
subregister above the threshold implies its main register ("HI re"); hybrids
count towards each of their registers, so a period's shares add up to more
than 100% (`classifier_common.register_labels`). `--level sub` gives the full
labels (subregisters) instead of the nine main registers.

### The trained runs

All with `python 6_train_classifier.py hplt.dated.jsonl --model <model> ... --results runs/<name>`,
where 150M = `jhu-clsp/ettin-encoder-150m` and 400M = `jhu-clsp/ettin-encoder-400m`.
"all" below means `--crawl-month --collection --text-years 5`.

| Run | Model | Options |
|---|---|---|
| `150m_meta_startend` | 150M | `--text start_end` |
| `150m_meta_start` | 150M | `--text start` |
| `150m_meta_only` | 150M | `--text none` |
| `150m_text_start` | 150M | `--no-metadata --text start` |
| `150m_meta_start_1024` | 150M | `--text start --max-len 1024 --batch 16` (killed before saving; scored with `--trained runs/150m_meta_start_1024/checkpoints/checkpoint-15618 --text start --max-len 1024 --batch 16 --results runs/test_150m_meta_start_1024 --evaluate-test`) |
| `150m_meta_start_month` | 150M | `--text start --crawl-month` |
| `150m_meta_start_collection` | 150M | `--text start --collection` |
| `150m_meta_start_textyears` | 150M | `--text start --text-years 5` |
| `150m_meta_start_all` | 150M | `--text start` + all |
| `150m_meta_start_all_sigma05`, `_sigma10` | 150M | `--text start` + all + `--label-sigma 0.5` / `1.0` |
| `curve_20k`, `curve_40k` | 150M | `--text start` + all + `--train-limit 20000` / `40000` |
| `150m_meta_only_rich` | 150M | `--text none --crawl-month --collection` |
| `400m_meta_start` | 400M | `--text start` |
| `400m_meta_start_all` | 400M | `--text start` + all — **the final model** |
| `400m_meta_only_rich` | 400M | `--text none --crawl-month --collection` |
| `logreg` | — | `7_logreg_baseline.py hplt.dated.jsonl --results runs/logreg` (C chosen from 1, 4, 16 on dev); test: `--results runs/test_logreg --evaluate-test` |

Defaults: 512 tokens, 3 epochs, batch 32, learning rate 3e-5, best epoch by
dev exact accuracy, split seed 42, bfloat16. Test scores of all runs:
`./score_all.sh` (uses `--trained`, so no retraining).
