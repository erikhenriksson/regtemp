"""Score htmldate and the classifier against publication years found by hand.

Reads the sheet from 14_annotation_sample.py after it has been filled in,
and its key. Documents whose manual_year is empty, or whose date_type is
"none", have no date a person could find; they are counted but not scored.

Prints, overall and per stratum register and period:

    found        share of documents with a manual year
    htmldate     share whose htmldate year equals the manual year
    classifier   the same for the classifier's most likely year (and
                 within one year)
    updated      share of the dated documents whose date is an update date
                 (date_type "updated"), where htmldate may be misled

"Overall" is weighted by the size of each register and period stratum among
the held-out documents, so that it estimates the accuracy over all of them;
"unweighted" treats every sampled document alike. Also prints how often the
register labels were judged correct (register_ok), by period.

    python 15_score_annotation.py runs/annotation_sheet.csv runs/annotation_key.csv
"""

import argparse

import numpy as np
import pandas as pd


def summary(g, weights=None):
    w = np.ones(len(g)) if weights is None else weights
    dated = g.dated.to_numpy()
    row = {"documents": len(g), "found": np.average(dated, weights=w)}
    if dated.any():
        wd = w[dated]
        d = g[dated]
        row["htmldate"] = np.average(d.htmldate_ok, weights=wd)
        if "classifier_ok" in d:
            row["classifier"] = np.average(d.classifier_ok, weights=wd)
            row["classifier_1"] = np.average(d.classifier_1, weights=wd)
        row["updated"] = np.average(d.date_type.eq("updated"), weights=wd)
    return row


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("sheet", help="the filled-in sheet")
    ap.add_argument("key", help="the key from 14_annotation_sample.py")
    args = ap.parse_args()

    sheet = pd.read_csv(args.sheet, encoding="utf-8-sig", dtype=str).fillna("")
    key = pd.read_csv(args.key, keep_default_na=False)
    df = key.merge(
        sheet[["item", "manual_year", "date_type", "register_ok"]].assign(
            item=lambda s: s.item.astype(int)
        ),
        on="item",
    )
    df["date_type"] = df.date_type.str.strip().str.lower()
    year = pd.to_numeric(df.manual_year.str.strip().str[:4], errors="coerce")
    df["dated"] = year.notna() & df.date_type.ne("none")
    df["htmldate_ok"] = df.pub_date.astype(str).str[:4].astype(int) == year
    if "pred_model" in df:
        pred = pd.to_numeric(df.pred_model, errors="coerce")
        df["classifier_ok"] = pred == year
        df["classifier_1"] = (pred - year).abs() <= 1

    # weight = stratum size / documents sampled from the stratum
    cell = df.groupby(["stratum_register", "stratum_period"]).item.transform("size")
    weights = (df.stratum_size / cell).to_numpy(float)

    rows = [{"group": "overall (weighted)"} | summary(df, weights)]
    rows.append({"group": "unweighted"} | summary(df))
    for col in ["stratum_register", "stratum_period"]:
        for name, g in df.groupby(col):
            rows.append({"group": f"{col.split('_')[1]} {name}"} | summary(g))
    pd.set_option("display.width", 160)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    judged = df[df.register_ok.str.strip().str.lower().isin(["yes", "no"])]
    if len(judged):
        ok = judged.register_ok.str.strip().str.lower().eq("yes")
        print("\nRegister labels judged correct, by period:")
        print(
            ok.groupby(judged.stratum_period).agg(["mean", "size"]).round(3).to_string()
        )


if __name__ == "__main__":
    main()
