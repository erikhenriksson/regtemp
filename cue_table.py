"""Compare the classifier with the explicit-date rule and the crawl year, by date cue.

Reads the test predictions of a run of 6_train_classifier.py and the texts
from the dated documents. Each test document gets a date cue: whether its
URL contains a year (the classifier's URL rule in classifier_common.py),
whether its text mentions a year (1950 up to the crawl year), both, or
neither. For each cue, and for all documents, prints exact-year accuracy,
within-one-year accuracy and mean absolute error of:

    model        the classifier's most likely year
    rule         the URL year if there is one, otherwise the most recent year
                 mentioned in the text, otherwise the crawl year
    crawl_year   always the crawl year

This gives the paper's table of results by date cue, and the "rule" and
"always the crawl year" rows of the main results table.

    python cue_table.py runs/test_400m_meta_start_all/predictions_test.csv \\
        hplt.dated.jsonl --out runs/rule_vs_model.csv
"""

import argparse
import json
import re

import pandas as pd

YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")
CUES = ["URL and text", "URL only", "text only", "neither"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("predictions", help="predictions_test.csv of a run")
    ap.add_argument("data", help="output of 3_date_html.py, for the texts")
    ap.add_argument("--out", help="write the table to this CSV")
    args = ap.parse_args()

    d = pd.read_csv(args.predictions)
    wanted = set(d.id)
    texts = {}
    for line in open(args.data, encoding="utf-8"):
        doc = json.loads(line)
        if doc["id"] in wanted:
            texts[doc["id"]] = doc.get("text") or ""

    latest = [
        max((int(y) for y in YEAR.findall(texts[i]) if int(y) <= c), default=None)
        for i, c in zip(d.id, d.crawl_year)
    ]
    d["text_year"] = pd.Series(latest, dtype="float")
    d["rule"] = d.pred_url.fillna(d.text_year).fillna(d.crawl_year)
    url, text = d.pred_url.notna(), d.text_year.notna()
    d["cue"] = [
        CUES[0] if u and t else CUES[1] if u else CUES[2] if t else CUES[3]
        for u, t in zip(url, text)
    ]

    rows = []
    for name, g in [*((c, d[d.cue == c]) for c in CUES), ("all", d)]:
        row = {"cue": name, "n": len(g), "share": len(g) / len(d)}
        for method, col in [
            ("model", "pred_model"),
            ("rule", "rule"),
            ("crawl_year", "pred_crawl_year"),
        ]:
            err = g[col] - g.year
            row[f"{method}_exact"] = (err == 0).mean()
            row[f"{method}_within_1"] = (err.abs() <= 1).mean()
            row[f"{method}_mae"] = err.abs().mean()
        rows.append(row)
    table = pd.DataFrame(rows).set_index("cue")

    pd.set_option("display.width", 220)
    print(table.round(3).to_string())
    if args.out:
        table.round(4).to_csv(args.out)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
