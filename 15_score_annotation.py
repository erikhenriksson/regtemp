"""Score htmldate and the classifier against publication years found by hand.

Reads the sheet from 14_annotation_sample.py after it has been filled in,
and its key. Rows with neither a manual_year nor a date_type are not yet
annotated and are left out. Documents whose date_type is "none" have no
date a person could find; they are counted but not scored.

Everything is printed twice: with all manual years, and with copyright
years (date_type "copyright") counted as no date, because a copyright
footer usually gives the year the page was served, not written.

Prints, overall and per stratum register and period:

    found        share of documents with a manual year
    htmldate     share whose htmldate year equals the manual year (and
                 within one year)
    classifier   the same for the classifier's most likely year
    crawl        the same for the crawl year
    updated      share of the dated documents whose date is an update date
                 (date_type "updated"), where htmldate may be misled

The overall weighted rows get 95% intervals from a bootstrap that resamples
documents within each register and period cell (1,000 resamples). Then
htmldate and classifier accuracy by date_type.

"Overall" is weighted by the size of each register and period stratum among
the held-out documents, so that it estimates the accuracy over all of them;
"unweighted" treats every sampled document alike. Also prints how often the
register labels were judged correct (register_ok), by period.

    python 15_score_annotation.py runs/annotation_sample_sheet_196.csv runs/annotation_key.csv
"""

import argparse

import numpy as np
import pandas as pd

METHODS = {"htmldate": "pub_year", "classifier": "pred_year", "crawl": "crawl_year"}


def summary(g, weights=None):
    w = np.ones(len(g)) if weights is None else np.asarray(weights, float)
    dated = g.dated.to_numpy()
    row = {"documents": len(g), "found": np.average(dated, weights=w)}
    if dated.any():
        wd = w[dated]
        d = g[dated]
        for name, col in METHODS.items():
            diff = (d[col] - d.year).abs()
            row[name] = np.average(diff.eq(0), weights=wd)
            row[name + "_1"] = np.average(diff.le(1), weights=wd)
        row["updated"] = np.average(d.date_type.eq("updated"), weights=wd)
    return row


def bootstrap(df, weights, n_boot, rng):
    """95% intervals for the weighted overall row, resampling within cells."""
    cells = df.groupby(["stratum_register", "stratum_period"]).indices.values()
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in cells])
        rows.append(summary(df.iloc[idx], weights[idx]))
    b = pd.DataFrame(rows).drop(columns="documents")
    return b.quantile(0.025), b.quantile(0.975)


def report(df, weights, n_boot, rng):
    rows = [{"group": "overall (weighted)"} | summary(df, weights)]
    low, high = bootstrap(df, weights, n_boot, rng)
    rows.append({"group": "  95% low"} | low.to_dict())
    rows.append({"group": "  95% high"} | high.to_dict())
    rows.append({"group": "unweighted"} | summary(df))
    for col in ["stratum_register", "stratum_period"]:
        for name, g in df.groupby(col):
            rows.append({"group": f"{col.split('_')[1]} {name}"} | summary(g))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("sheet", help="the filled-in sheet")
    ap.add_argument("key", help="the key from 14_annotation_sample.py")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    sheet = pd.read_csv(args.sheet, encoding="utf-8-sig", dtype=str).fillna("")
    key = pd.read_csv(args.key, keep_default_na=False)
    df = key.merge(
        sheet[["item", "manual_year", "date_type", "register_ok"]].assign(
            item=lambda s: s.item.astype(int)
        ),
        on="item",
    )
    df["date_type"] = df.date_type.str.strip().str.lower()
    df["year"] = pd.to_numeric(df.manual_year.str.strip().str[:4], errors="coerce")
    annotated = df.year.notna() | df.date_type.ne("")
    print(f"{annotated.sum()} of {len(df)} documents annotated\n")
    df = df[annotated].reset_index(drop=True)
    df["pub_year"] = df.pub_date.astype(str).str[:4].astype(int)
    df["pred_year"] = pd.to_numeric(df.pred_model, errors="coerce")
    df["crawl_year"] = df.crawl_date.astype(str).str[:4].astype(int)

    # weight = stratum size / documents sampled from the stratum
    cell = df.groupby(["stratum_register", "stratum_period"]).item.transform("size")
    weights = (df.stratum_size / cell).to_numpy(float)
    pd.set_option("display.width", 200)

    print("All manual years:")
    df["dated"] = df.year.notna() & df.date_type.ne("none")
    report(df, weights, args.n_boot, rng)

    print("\nCopyright years counted as no date:")
    df["dated"] = df.year.notna() & ~df.date_type.isin(["none", "copyright"])
    report(df, weights, args.n_boot, rng)

    print("\nAccuracy by date type (unweighted):")
    d = df[df.year.notna()]
    by_type = d.groupby(d.date_type.replace("", "(plain)")).apply(
        lambda g: pd.Series(
            {"documents": len(g)}
            | {m: (g[c] == g.year).mean() for m, c in METHODS.items()}
        )
    )
    print(by_type.round(3).to_string())

    judged = df[df.register_ok.str.strip().str.lower().isin(["yes", "no"])]
    if len(judged):
        ok = judged.register_ok.str.strip().str.lower().eq("yes")
        print("\nRegister labels judged correct, by period:")
        print(
            ok.groupby(judged.stratum_period).agg(["mean", "size"]).round(3).to_string()
        )


if __name__ == "__main__":
    main()
