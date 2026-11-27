"""How old are the documents of the sample: publication years and age at crawl.

From the classifier's predictions for the whole sample (9_predict_years.py),
and optionally from htmldate's dates for the dated documents, computes:

    year_distribution.csv   the share of documents published in each year,
                            for all documents and per crawl year
    age_at_crawl.csv        the share of documents that are 0, 1, 2, 3-5 and
                            6 or more years old when crawled (crawl year minus
                            publication year), and the median age, for all
                            documents and per main register, over all crawls
                            and per crawl year
    years.png, years.pdf    the publication-year distribution of the sample,
                            with that of the htmldate-dated documents

With the classifier, each document counts in every year in proportion to
its predicted probability (summed probabilities). Registers are main
registers at probability 0.4 or more, a subregister implying its main
register, and a hybrid counting towards each (classifier_common.TAXONOMY);
"all" counts every document once. Years before --min-year count as
--min-year, as in training.

Age at crawl mixes several things: how long pages stay online, how much of
a register was published in each year, which crawls the sample comes from,
and what the date means. Comparing within one crawl year removes the third.

    python 12_age_of_sample.py --predictions runs/predict_400m_meta_start_all \\
        --dated $DATA/hplt.dated.jsonl --results runs/age
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

import classifier_common as common

AGE_BINS = {"0": (0, 0), "1": (1, 1), "2": (2, 2), "3-5": (3, 5), "6+": (6, 999)}
PERIODS = [(1995, 2005), (2006, 2010), (2011, 2015), (2016, 2019), (2020, 2025)]


def main_register_matrix(scores, threshold):
    """documents x main registers: 1 where the document has the register (a
    subregister above the threshold implies its main register)."""
    member = np.zeros((len(scores), len(common.MAIN_REGISTERS)), dtype=bool)
    for j, (main, subs) in enumerate(common.TAXONOMY.items()):
        cols = [c for c in [main] + subs if c in scores]
        member[:, j] = (scores[cols].fillna(0).to_numpy() >= threshold).any(axis=1)
    return member


def load_classifier(folder, docs, threshold):
    df = pd.read_csv(os.path.join(folder, "predictions.csv"))
    probs = np.load(os.path.join(folder, "probabilities.npy"))
    years = np.array(json.load(open(os.path.join(folder, "years.json"))))
    if docs == "unlabelled":
        keep = ~df.labelled.to_numpy()
        df, probs = df[keep].reset_index(drop=True), probs[keep]
    scores = df.rename(columns={f"reg_{r}": r for r in common.ALL_REGISTERS})
    member = main_register_matrix(scores, threshold)
    return df.crawl_year.to_numpy(), probs, years, member


def load_dated(path, years, min_year, threshold):
    """htmldate years as one-hot 'probabilities' over the classifier's years."""
    rows = []
    for line in open(path, encoding="utf-8"):
        doc = json.loads(line)
        if not doc.get("pub_date"):
            continue
        year = max(int(doc["pub_date"][:4]), min_year)
        crawl_year = int(doc["crawl_date"][:4])
        if year > crawl_year or year > years[-1]:
            continue  # after the crawl, as in classifier_common.load
        scores = doc.get("web-register") or {}
        rows.append({"year": year, "crawl_year": crawl_year} | scores)
    df = pd.DataFrame(rows)
    onehot = np.zeros((len(df), len(years)), dtype=np.float32)
    onehot[np.arange(len(df)), np.searchsorted(years, df.year)] = 1
    member = main_register_matrix(df, threshold)
    return df.crawl_year.to_numpy(), onehot, member


def summarise(source, crawl_years, probs, years, member):
    """Rows of year_distribution and age_at_crawl for one source."""
    year_rows, age_rows = [], []
    groups = [("all", np.ones(len(crawl_years), dtype=bool))] + [
        (str(c), crawl_years == c) for c in sorted(set(crawl_years))
    ]
    for group, rows in groups:
        mass = probs[rows].sum(0, dtype=np.float64)
        for y, m in zip(years, mass):
            year_rows.append(
                {
                    "source": source,
                    "crawl_year": group,
                    "year": int(y),
                    "documents": m,
                    "share": m / rows.sum(),
                }
            )
        for register, j in [("all", None)] + [
            (r, j) for j, r in enumerate(common.MAIN_REGISTERS)
        ]:
            sel = rows if j is None else rows & member[:, j]
            if not sel.any():
                continue
            # age histogram: per crawl year, shift the year mass by the crawl year
            age_mass = {}
            for c in sorted(set(crawl_years[sel])):
                m = probs[sel & (crawl_years == c)].sum(0, dtype=np.float64)
                for a, v in zip(c - years, m):
                    if a >= 0:
                        age_mass[a] = age_mass.get(a, 0) + v
            ages = np.array(sorted(age_mass))
            mass = np.array([age_mass[a] for a in ages])
            cum = mass.cumsum() / mass.sum()
            row = {
                f"share_{name}": mass[(ages >= lo) & (ages <= hi)].sum() / mass.sum()
                for name, (lo, hi) in AGE_BINS.items()
            }
            age_rows.append(
                {
                    "source": source,
                    "crawl_year": group,
                    "register": register,
                    "documents": int(sel.sum()),
                }
                | row
                | {
                    "median_age": int(ages[np.searchsorted(cum, 0.5)]),
                    "mean_age": float((ages * mass).sum() / mass.sum()),
                }
            )
    return year_rows, age_rows


def plot_years(year_dist, out):
    """Share of documents per publication year: classifier (whole sample) as
    bars, htmldate (dated documents) as a line."""
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    d = year_dist[year_dist.crawl_year == "all"]
    c = d[d.source == "classifier"]
    ax.bar(
        c.year, c.share, color="#257de0", width=0.8, label="whole sample, classifier"
    )
    h = d[d.source == "htmldate"]
    if len(h):
        ax.plot(
            h.year, h.share, color="black", lw=1.5, label="dated documents, htmldate"
        )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#e5e5e5")
    ax.tick_params(colors="#555555", labelsize=7, length=0)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out, f"years.{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--predictions", required=True, help="folder of 9_predict_years.py")
    ap.add_argument("--dated", help="output of 3_date_html.py, for htmldate's dates")
    ap.add_argument("--results", required=True)
    ap.add_argument(
        "--docs",
        choices=["sample", "unlabelled"],
        default="sample",
        help="all predicted documents, or only those outside the labelled lines",
    )
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--min-year", type=int, default=1995)
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)

    crawl, probs, years, member = load_classifier(
        args.predictions, args.docs, args.threshold
    )
    if not len(crawl):
        raise SystemExit(f"no documents with --docs {args.docs}")
    year_rows, age_rows = summarise("classifier", crawl, probs, years, member)
    print(f"classifier: {len(crawl)} documents ({args.docs})")
    if args.dated:
        crawl_h, onehot, member_h = load_dated(
            args.dated, years, args.min_year, args.threshold
        )
        y, a = summarise("htmldate", crawl_h, onehot, years, member_h)
        year_rows += y
        age_rows += a
        print(f"htmldate: {len(crawl_h)} dated documents")
    year_dist, age = pd.DataFrame(year_rows), pd.DataFrame(age_rows)
    year_dist.to_csv(os.path.join(args.results, "year_distribution.csv"), index=False)
    age.to_csv(os.path.join(args.results, "age_at_crawl.csv"), index=False)
    plot_years(year_dist, args.results)

    pd.set_option("display.width", 160)
    d = year_dist[year_dist.crawl_year == "all"]
    print("\nShare of documents by publication period (%):")
    print(
        pd.DataFrame(
            {
                s: {
                    f"{lo}-{hi}": 100 * g[g.year.between(lo, hi)].share.sum()
                    for lo, hi in PERIODS
                }
                for s, g in d.groupby("source")
            }
        )
        .round(1)
        .to_string()
    )
    cols = ["source", "register", "documents"] + [f"share_{b}" for b in AGE_BINS]
    cols += ["median_age", "mean_age"]
    shown = age[age.crawl_year == "all"][cols].copy()
    for b in AGE_BINS:
        shown[f"share_{b}"] = (100 * shown[f"share_{b}"]).round(1)
    print("\nAge at crawl, all crawls (% of documents; years):")
    print(shown.round(2).to_string(index=False))
    by_crawl = age[(age.register == "all") & (age.crawl_year != "all")][
        cols[:1] + ["crawl_year"] + cols[2:]
    ].copy()
    for b in AGE_BINS:
        by_crawl[f"share_{b}"] = (100 * by_crawl[f"share_{b}"]).round(1)
    print("\nAge at crawl by crawl year, all documents:")
    print(by_crawl.round(2).to_string(index=False))
    print("\nMedian publication year and share published up to 2015:")
    for source, g in d.groupby("source"):
        g = g.sort_values("year")
        median = int(g.year.iloc[(g.share.cumsum() >= 0.5).argmax()])
        early = 100 * g[g.year <= 2015].share.sum()
        print(f"  {source}: median {median}, up to 2015 {early:.1f}%")
    within = age[(age.source == "classifier") & (age.crawl_year != "all")]
    print("\nShare six or more years old (%), by register and crawl year (classifier):")
    print(
        within.pivot(index="register", columns="crawl_year", values="share_6+")
        .mul(100)
        .round(1)
        .to_string()
    )
    print(f"\nAll results in {args.results}/")


if __name__ == "__main__":
    main()
