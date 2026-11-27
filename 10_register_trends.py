"""Shares of the main registers by publication period.

For a set of documents, counts how the main registers (MT, LY, SP, ID, NA,
HI, IN, OP, IP) are spread over publication periods, with the publication
year from htmldate, from the date classifier, or both. Each register's share
in a period is the share of that period's documents that have the register.

Documents (--docs):

    dated       all documents htmldate dated (from --dated)
    dev, test   the dated documents of one split; heldout is both. With
    heldout     --predictions, these documents are dated by htmldate and by
                the classifier, which isolates the effect of the dating method
    sample      every document in --predictions (output of 9_predict_years.py)
    unlabelled  the same without the htmldate-labelled lines

Dating methods, whichever the documents allow:

    htmldate        htmldate's fast-mode year (the labels)
    model           the classifier's most likely year
    model_median    the median of the predicted distribution
    model_prob      the predicted probabilities: a document counts in every
                    period, in proportion to its probability for that period

A document's register (--registers): argmax, the main register with the
highest probability (classifier_common.main_register; shares add up to 100%),
or threshold, every main register whose probability is at least --threshold
(a document can have several or none; "(none)" is the share with none).

Years are clipped below at --min-year, as in training, and htmldate years
after the crawl year are left out, as in classifier_common.load. The 95%
intervals come from a Poisson bootstrap over documents: they show sampling
uncertainty, not the classifier's dating error.

Prints a table per method (shares in %, rows = periods), and with htmldate
and a model method on the same documents, their difference in percentage
points. Writes --out, one row per method, period and register: documents
(the period's number of documents, summed probability for model_prob),
share, low and high.

    python 10_register_trends.py --dated hplt.dated.jsonl --docs dated \\
        --out runs/registers_dated.csv
    python 10_register_trends.py --dated hplt.dated.jsonl \\
        --predictions runs/predict_400m_meta_start_all --docs heldout \\
        --out runs/registers_heldout.csv
    python 10_register_trends.py --predictions runs/predict_400m_meta_start_all \\
        --docs unlabelled --out runs/registers_unlabelled.csv
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

import classifier_common as common

REGISTERS = common.MAIN_REGISTERS
DEFAULT_PERIODS = "1995-2005,2006-2010,2011-2015,2016-2019,2020-2025"


def parse_periods(spec):
    """'1995-2005,2006' -> [('1995-2005', 1995, 2005), ('2006', 2006, 2006)]"""
    periods = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        periods.append((part, int(lo), int(hi or lo)))
    return periods


def period_matrix(years, periods):
    """years x periods: 1 where the year falls in the period."""
    years = np.asarray(years)
    return np.stack([(years >= lo) & (years <= hi) for _, lo, hi in periods], 1)


def load_dated(path, seed, min_year):
    rows, too_late = [], 0
    for line in open(path, encoding="utf-8"):
        doc = json.loads(line)
        if not doc.get("pub_date"):
            continue
        year = max(int(doc["pub_date"][:4]), min_year)
        crawl_year = int(doc["crawl_date"][:4])
        if year > crawl_year:
            too_late += 1
            continue
        scores = doc.get("web-register") or {}
        rows.append(
            {"id": doc["id"], "year": year, "split": common.split_of(doc["id"], seed)}
            | {r: scores.get(r, np.nan) for r in REGISTERS}
        )
    print(f"{len(rows)} dated documents ({too_late} dated after their crawl left out)")
    return pd.DataFrame(rows)


def load_predictions(folder):
    df = pd.read_csv(os.path.join(folder, "predictions.csv"))
    df = df.rename(columns={f"reg_{r}": r for r in REGISTERS})
    df["row"] = np.arange(len(df))  # row in probabilities.npy
    probs = np.load(os.path.join(folder, "probabilities.npy"))
    years = np.array(json.load(open(os.path.join(folder, "years.json"))))
    return df, probs, years


def select(args):
    """The documents, and the period weights (documents x periods) per method."""
    periods = parse_periods(args.periods)
    pred = probs = years = None
    if args.predictions:
        pred, probs, years = load_predictions(args.predictions)
    if args.docs in ("dated", "dev", "test", "heldout"):
        if not args.dated:
            raise SystemExit(f"--docs {args.docs} needs --dated")
        df = load_dated(args.dated, args.seed, args.min_year)
        if args.docs != "dated":
            keep = ["dev", "test"] if args.docs == "heldout" else [args.docs]
            df = df[df.split.isin(keep)]
        if pred is not None:
            n = len(df)
            df = df.merge(
                pred[["id", "row", "pred_model", "pred_model_median"]], on="id"
            )
            if len(df) < n:
                print(f"{n - len(df)} dated documents have no prediction, left out")
            if (df.split == "train").any():
                print(
                    "note: the model methods include the classifier's training documents"
                )
    else:
        if pred is None:
            raise SystemExit(f"--docs {args.docs} needs --predictions")
        df = pred if args.docs == "sample" else pred[~pred.labelled]
    no_scores = df[REGISTERS].isna().all(axis=1)
    if no_scores.any():
        print(f"{int(no_scores.sum())} documents without register scores, left out")
    df = df[~no_scores].reset_index(drop=True)
    df[REGISTERS] = df[REGISTERS].fillna(0)

    weights = {}
    if "year" in df:
        weights["htmldate"] = period_matrix(df.year, periods).astype(float)
    if pred is not None:
        for m in ("model", "model_median"):
            year = df[f"pred_{m}"].clip(lower=args.min_year)
            weights[m] = period_matrix(year, periods).astype(float)
        weights["model_prob"] = probs[df.row.to_numpy()] @ period_matrix(
            np.maximum(years, args.min_year), periods
        )
    return df, weights, periods


def membership(df, how, threshold):
    """documents x registers: 1 where the document has the register."""
    scores = df[REGISTERS].to_numpy(float)
    if how == "argmax":
        member = np.zeros_like(scores)
        member[np.arange(len(scores)), scores.argmax(1)] = 1
        return member, REGISTERS
    member = (scores >= threshold).astype(float)
    none = (member.sum(1) == 0).astype(float)[:, None]
    return np.hstack([member, none]), REGISTERS + ["(none)"]


def shares(w, member, v=None):
    """periods x registers: weighted share of each period's documents."""
    if v is not None:
        w = w * v[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w.T @ member) / w.sum(0)[:, None]


def trends(w, member, n_boot, rng):
    point = shares(w, member)
    if not n_boot:
        return point, point * np.nan, point * np.nan
    boot = np.stack(
        [shares(w, member, rng.poisson(1.0, len(w))) for _ in range(n_boot)]
    )
    with warnings.catch_warnings():  # periods without documents
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return point, low, high


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dated", help="output of 3_date_html.py (hplt.dated.jsonl)")
    ap.add_argument("--predictions", help="results folder of 9_predict_years.py")
    ap.add_argument(
        "--docs",
        required=True,
        choices=["dated", "dev", "test", "heldout", "sample", "unlabelled"],
    )
    ap.add_argument("--out", required=True, help="CSV of all shares and intervals")
    ap.add_argument("--registers", choices=["argmax", "threshold"], default="argmax")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--periods", default=DEFAULT_PERIODS)
    ap.add_argument("--min-year", type=int, default=1995)
    ap.add_argument("--bootstrap", type=int, default=200, help="0: no intervals")
    ap.add_argument("--seed", type=int, default=42, help="seed of the split")
    args = ap.parse_args()

    df, weights, periods = select(args)
    member, labels = membership(df, args.registers, args.threshold)
    names = [name for name, _, _ in periods]
    print(f"{len(df)} documents ({args.docs}), registers by {args.registers}")
    rng = np.random.default_rng(0)
    rows, tables = [], {}
    pd.set_option("display.width", 160)
    for method, w in weights.items():
        point, low, high = trends(w, member, args.bootstrap, rng)
        docs = w.sum(0)
        outside = len(df) - docs.sum()
        for i, period in enumerate(names):
            for j, register in enumerate(labels):
                rows.append(
                    {
                        "docs": args.docs,
                        "method": method,
                        "registers": args.registers,
                        "period": period,
                        "register": register,
                        "documents": docs[i],
                        "share": point[i, j],
                        "low": low[i, j],
                        "high": high[i, j],
                    }
                )
        table = pd.DataFrame(100 * point, index=names, columns=labels)
        tables[method] = table
        print(f"\n{method}: share of each period's documents (%)")
        print(table.assign(documents=docs.round()).round(1).to_string())
        if outside > 0.5:
            print(f"({outside:.0f} documents fall outside the periods)")
    if "htmldate" in tables:
        for method in [m for m in tables if m != "htmldate"]:
            diff = tables[method] - tables["htmldate"]
            print(f"\n{method} minus htmldate (percentage points)")
            print(diff.round(1).to_string())
            print(f"mean absolute difference: {np.nanmean(np.abs(diff.values)):.2f}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
