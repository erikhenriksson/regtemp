"""Precision and recall of the classifier for placing documents into periods.

Reads the predictions of a run of 6_train_classifier.py. For each period,
prints how many documents are labelled with a year in it, how many the
classifier places there, and precision and recall with 95% Wilson intervals,
for all predictions and for confident ones (probability of the predicted
year at least --threshold). This gives the paper's period table.

    python period_table.py runs/test_400m_meta_start_all/predictions_test.csv \\
        --out runs/periods_test.csv
"""

import argparse
import math

import pandas as pd

PERIODS = [(1995, 2005), (2006, 2010), (2011, 2015), (2016, 2019), (2020, 2025)]


def wilson(k, n, z=1.96):
    """Proportion k/n with its 95% Wilson interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


def fmt(value):
    p, lo, hi = value
    return f"{p:.2f} ({lo:.2f}-{hi:.2f})"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("predictions", help="predictions_<split>.csv of a run")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="confidence threshold, chosen on the development set",
    )
    ap.add_argument("--out", help="write the table to this CSV")
    args = ap.parse_args()

    d = pd.read_csv(args.predictions)
    rows = []
    for lo, hi in PERIODS:
        true = d.year.between(lo, hi)
        pred = d.pred_model.between(lo, hi)
        conf = pred & (d.confidence >= args.threshold)
        rows.append(
            {
                "period": f"{lo}-{hi}",
                "truly": int(true.sum()),
                "predicted": int(pred.sum()),
                "precision": fmt(wilson((true & pred).sum(), pred.sum())),
                "recall": fmt(wilson((true & pred).sum(), true.sum())),
                "confident": int(conf.sum()),
                "confident precision": fmt(wilson((true & conf).sum(), conf.sum())),
                "confident recall": f"{(true & conf).sum() / max(true.sum(), 1):.2f}",
            }
        )
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
