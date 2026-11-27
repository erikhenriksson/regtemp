#!/bin/bash
# Score the test split for every trained model in runs/ that has no test
# results yet, then print dev and test results for all of them, and save
# them to runs/all_results.csv. This gives the paper's main results table
# and the appendix table of all runs (with cue_table.py for the rule and
# crawl-year rows).
#
#     ./score_all.sh              (run on a GPU node, from the data folder)
#
# TRAIN can name another copy of the training script: TRAIN=6_train_classifier3.py ./score_all.sh
# Models evaluated from a checkpoint (the 1,024-token run, killed before it
# saved best_model) have to be scored by hand with --trained <checkpoint>.

TRAIN=${TRAIN:-6_train_classifier.py}
D=hplt.dated.jsonl
for dir in runs/*/; do
  r=$(basename "$dir")
  [[ $r == test_* ]] && continue
  if [[ ! -d runs/$r/best_model ]]; then echo "skipping $r: no best_model"; continue; fi
  if [[ -f runs/test_$r/metrics.csv ]]; then echo "already scored: $r"; continue; fi
  echo "=== scoring $r"
  python "$TRAIN" $D --trained runs/$r/best_model --results runs/test_$r --evaluate-test
done
[[ -f runs/test_logreg/metrics.csv ]] || python 7_logreg_baseline.py $D --results runs/test_logreg --evaluate-test

python - <<'PY'
import glob, pandas as pd
rows = []
for f in sorted(glob.glob("runs/test_*/metrics.csv")):
    m = pd.read_csv(f)
    m = m[(m.subset == "all") & (m.method == "model")]
    for _, r in m.iterrows():
        rows.append({"run": f.split("/")[1][5:], "split": r.split, "exact": r.exact,
                     "within_1": r.within_1, "mae": r.mae, "bias": r.bias})
t = pd.DataFrame(rows).pivot(index="run", columns="split", values=["exact", "within_1", "mae", "bias"])
t = t.sort_values(("exact", "dev"), ascending=False).round(3)
pd.set_option("display.width", 200)
print(t.to_string())
t.to_csv("runs/all_results.csv")
print("\nSaved to runs/all_results.csv")
PY
