"""Shares of the registers by publication period.

For a set of documents, counts how the registers are spread over
publication periods, with the publication year from htmldate, from the date
classifier, or both. Each register's share in a period is the share of that
period's documents that have the register.

Registers follow the two-level taxonomy of HPLT's web-register labels
(classifier_common.TAXONOMY and register_labels). A document has every label
whose probability is at least --threshold (0.4). A subregister above the
threshold is one label with its main register ("HI re"), and it implies the
main register even when the main register's own probability is lower.
Hybrids (such as "HI re" and "ID") count towards every register they have,
so the shares of a period add up to more than 100%; "(none)" is the share
of documents with no label above the threshold.

--level main (default) counts the nine main registers: NA covers "NA",
"NA ne", "NA sr" and "NA nb", and a document with "NA ne" and "NA nb"
counts once. --level sub counts the full labels: there, "NA" means a
narrative without a subregister above the threshold.

Documents (--docs):

    dated       all documents htmldate dated (from --dated)
    dev, test   the dated documents of one split; heldout is both. With
    heldout     --predictions, these documents are dated by htmldate and by
                the classifier, which isolates the effect of the dating method
    sample      every document in --predictions (output of 9_predict_years.py)
    unlabelled  the same without the htmldate-labelled lines

Dating methods, whichever the documents allow:

    htmldate        htmldate's fast-mode year (the labels)
    crawl           the crawl year, as if documents were published when crawled
                    (what studies get that date documents by their crawl)
    model           the classifier's most likely year
    model_median    the median of the predicted distribution
    model_prob      the predicted probabilities: a document counts in every
                    period, in proportion to its probability for that period

--run NAME=FOLDER (repeatable, with --docs dev, test or heldout) adds the
predictions of another trained model, from the results folder of
6_train_classifier.py or 7_logreg_baseline.py (predictions_<split>.csv,
probabilities_<split>.npy, years.json; run with --evaluate-test for the test
split): NAME (summed probabilities) and NAME_likely (most likely year). Only
documents that every run has predictions for are kept.

--periods takes periods such as "1995-2005,2006-2010", or "1995..2025" for
one period per year (for 11_plot_register_trends.py). Years are clipped
below at --min-year, as in training, and htmldate years
after the crawl year are left out, as in classifier_common.load. The 95%
intervals come from a Poisson bootstrap that resamples websites
(--bootstrap-by website, the default where the URLs are known, that is for
dated documents), so that documents from the same website count as one
cluster, or single documents (--bootstrap-by document; always for --docs
sample and unlabelled, whose URLs are not in the predictions). They show
sampling uncertainty, not the classifier's dating error.

Prints a table per method (shares in %, one column per period), and with
htmldate and a model method on the same documents, their difference in
percentage points. Writes --out, one row per method, period and register: documents
(the period's number of documents, summed probability for model_prob),
share, low and high. Read it with pandas.read_csv(..., keep_default_na=False),
or pandas turns the register NA into a missing value.

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

REGISTERS = common.ALL_REGISTERS  # the 25 labels of the web-register field
DEFAULT_PERIODS = "1995-2005,2006-2010,2011-2015,2016-2019,2020-2025"


def parse_periods(spec):
    """'1995-2005,2006' -> [('1995-2005', 1995, 2005), ('2006', 2006, 2006)];
    '1995..1997' -> one period per year: 1995, 1996, 1997."""
    periods = []
    for part in spec.split(","):
        if ".." in part:
            lo, hi = map(int, part.split(".."))
            periods += [(str(y), y, y) for y in range(lo, hi + 1)]
            continue
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
            {
                "id": doc["id"],
                "year": year,
                "crawl_year": crawl_year,
                "website": common.website(doc["u"]),
                "split": common.split_of(doc["id"], seed),
            }
            | {r: scores.get(r, np.nan) for r in REGISTERS}
        )
    print(f"{len(rows)} dated documents ({too_late} dated after their crawl left out)")
    return pd.DataFrame(rows)


def load_predictions(folder):
    df = pd.read_csv(os.path.join(folder, "predictions.csv"))
    df = df.rename(columns={f"reg_{r}": r for r in REGISTERS})
    df = df.reindex(columns=list(df.columns) + [r for r in REGISTERS if r not in df])
    df["row"] = np.arange(len(df))  # row in probabilities.npy
    probs = np.load(os.path.join(folder, "probabilities.npy"))
    years = np.array(json.load(open(os.path.join(folder, "years.json"))))
    return df, probs, years


def load_run(folder, splits):
    """ids and predicted probabilities of a training run for the given splits."""
    ids, probs = [], []
    for s in splits:
        csv = os.path.join(folder, f"predictions_{s}.csv")
        npy = os.path.join(folder, f"probabilities_{s}.npy")
        if not (os.path.exists(csv) and os.path.exists(npy)):
            raise SystemExit(f"{folder} has no predictions for {s} ({csv}, {npy})")
        ids += pd.read_csv(csv, usecols=["id"]).id.tolist()
        probs.append(np.load(npy))
    years = np.array(json.load(open(os.path.join(folder, "years.json"))))
    return (
        pd.DataFrame({"id": ids, "row": np.arange(len(ids))}),
        np.vstack(probs),
        years,
    )


def select(args):
    """The documents, and the period weights (documents x periods) per method."""
    periods = parse_periods(args.periods)
    pred = probs = years = None
    runs = {}  # --run: name -> (probabilities, years)
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
        for spec in args.run or []:
            name, _, folder = spec.partition("=")
            ids, run_probs, run_years = load_run(folder, sorted(df.split.unique()))
            n = len(df)
            df = df.merge(ids.rename(columns={"row": f"row_{name}"}), on="id")
            if len(df) < n:
                print(f"{name}: {n - len(df)} documents have no prediction, left out")
            runs[name] = (run_probs, run_years)
    else:
        if args.run:
            raise SystemExit("--run needs --docs dev, test or heldout")
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
    crawl = df.crawl_year.clip(lower=args.min_year)
    weights["crawl"] = period_matrix(crawl, periods).astype(float)
    if pred is not None:
        for m in ("model", "model_median"):
            year = df[f"pred_{m}"].clip(lower=args.min_year)
            weights[m] = period_matrix(year, periods).astype(float)
        weights["model_prob"] = probs[df.row.to_numpy()] @ period_matrix(
            np.maximum(years, args.min_year), periods
        )
    for name, (run_probs, run_years) in runs.items():
        p = run_probs[df[f"row_{name}"].to_numpy()]
        weights[name] = p @ period_matrix(np.maximum(run_years, args.min_year), periods)
        likely = np.maximum(run_years[p.argmax(1)], args.min_year)
        weights[f"{name}_likely"] = period_matrix(likely, periods).astype(float)
    return df, weights, periods


def membership(df, level, threshold):
    """documents x registers: 1 where the document has the register, and the
    share of documents that are hybrids (two or more main registers)."""
    labels = [
        common.register_labels(scores, threshold)
        for scores in df[REGISTERS].to_dict("records")
    ]
    mains = [{label.split()[0] for label in doc} for doc in labels]
    names = common.MAIN_REGISTERS if level == "main" else common.FINE_REGISTERS
    column = {name: j for j, name in enumerate(names + ["(none)"])}
    member = np.zeros((len(df), len(column)))
    for i, doc in enumerate(mains if level == "main" else labels):
        for label in doc or ["(none)"]:
            member[i, column[label]] = 1
    hybrid = np.mean([len(m) > 1 for m in mains])
    return member, list(column), hybrid


def shares(w, member, v=None):
    """periods x registers: weighted share of each period's documents."""
    if v is not None:
        w = w * v[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w.T @ member) / w.sum(0)[:, None]


def trends(w, member, n_boot, rng, clusters):
    """Shares and 95% intervals; clusters gives each document's cluster index
    (its website, or itself), and the bootstrap resamples clusters."""
    point = shares(w, member)
    if not n_boot:
        return point, point * np.nan, point * np.nan
    n_clusters = clusters.max() + 1
    boot = np.stack(
        [
            shares(w, member, rng.poisson(1.0, n_clusters)[clusters])
            for _ in range(n_boot)
        ]
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
    ap.add_argument("--level", choices=["main", "sub"], default="main")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--periods", default=DEFAULT_PERIODS)
    ap.add_argument("--min-year", type=int, default=1995)
    ap.add_argument("--bootstrap", type=int, default=200, help="0: no intervals")
    ap.add_argument(
        "--bootstrap-by",
        choices=["website", "document"],
        default="website",
        help="resample websites (where URLs are known) or single documents",
    )
    ap.add_argument("--seed", type=int, default=42, help="seed of the split")
    ap.add_argument(
        "--run",
        action="append",
        metavar="NAME=FOLDER",
        help="another trained model's dev/test predictions (repeatable)",
    )
    args = ap.parse_args()

    df, weights, periods = select(args)
    member, labels, hybrid = membership(df, args.level, args.threshold)
    names = [name for name, _, _ in periods]
    print(
        f"{len(df)} documents ({args.docs}), {args.level} registers at threshold "
        f"{args.threshold}; {hybrid:.1%} are hybrids (two or more main registers)"
    )
    if args.bootstrap_by == "website" and "website" in df:
        clusters = pd.factorize(df.website)[0]
        print(f"bootstrap over {clusters.max() + 1} websites")
    else:
        clusters = np.arange(len(df))
        print("bootstrap over documents")
    rng = np.random.default_rng(0)
    rows, tables = [], {}
    pd.set_option("display.width", 160)
    for method, w in weights.items():
        point, low, high = trends(w, member, args.bootstrap, rng, clusters)
        docs = w.sum(0)
        outside = len(df) - docs.sum()
        for i, period in enumerate(names):
            for j, register in enumerate(labels):
                rows.append(
                    {
                        "docs": args.docs,
                        "method": method,
                        "level": args.level,
                        "threshold": args.threshold,
                        "bootstrap_by": (
                            "website" if clusters.max() + 1 < len(df) else "document"
                        ),
                        "period": period,
                        "register": register,
                        "documents": docs[i],
                        "share": point[i, j],
                        "low": low[i, j],
                        "high": high[i, j],
                    }
                )
        table = pd.DataFrame(100 * point.T, index=labels, columns=names)
        tables[method] = table
        print(f"\n{method}: share of each period's documents (%)")
        shown = pd.concat(
            [
                table.round(1),
                pd.DataFrame([docs.round()], index=["documents"], columns=names),
            ]
        )
        print(shown.to_string())
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
