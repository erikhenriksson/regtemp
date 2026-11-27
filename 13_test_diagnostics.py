"""Check the classifier's test predictions for register bias and calibration.

Compares the classifier's predictions (9_predict_years.py) with htmldate's
years on the documents of one split (test by default), and writes into
--results:

    by_register.csv     per main register (a hybrid counts towards each;
                        classifier_common.main_registers) and for all
                        documents: number of documents, exact-year accuracy,
                        and the bias (predicted minus htmldate year, > 0 is
                        too late) of the most likely year, the median year,
                        the expected year (mean of the predicted
                        distribution) and the crawl year; and the expected
                        calibration error. A classifier that dated documents
                        by their register would show register-specific bias.
                        Also the share of documents whose most likely year is
                        two or more years too early or too late, and how well
                        the register's whole year distribution is recovered:
                        total variation distance between its htmldate years
                        and the summed probabilities, the most likely years
                        and the crawl years (tvd_*), and the largest gap
                        between the cumulative distributions (ks_*: the
                        share published in year t or earlier, at the worst
                        t).
    reliability.csv     per band of confidence (probability of the most
                        likely year): documents, mean confidence, accuracy
    year_distribution.csv  share of the documents in each year: htmldate,
                        summed probabilities, most likely year
    cumulative.csv      per year t: the share of documents published in t or
                        earlier, by htmldate and by the summed probabilities
                        (calibration of the whole distribution, not only of
                        the most likely year)
    calibration.json    expected calibration error (15 equal-width bands),
                        Brier score, ranked probability score (of the
                        predicted distribution and of the crawl year, scaled
                        to 0-1), and the total variation distance between the
                        htmldate year distribution and the summed
                        probabilities (and the most likely years)

    python 13_test_diagnostics.py --predictions runs/predict_400m_meta_start_all \\
        --dated $DATA/hplt.dated.jsonl --results runs/test_diagnostics
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import classifier_common as common

BANDS = 15


def load(args):
    """The split's documents with htmldate year, crawl year, registers and
    predicted probabilities."""
    rows = []
    for line in open(args.dated, encoding="utf-8"):
        doc = json.loads(line)
        if not doc.get("pub_date"):
            continue
        if common.split_of(doc["id"], args.seed) != args.split:
            continue
        year = max(int(doc["pub_date"][:4]), args.min_year)
        crawl_year = int(doc["crawl_date"][:4])
        if year > crawl_year:
            continue  # as in classifier_common.load
        rows.append(
            {
                "id": doc["id"],
                "year": year,
                "crawl_year": crawl_year,
                "registers": common.main_registers(doc.get("web-register")),
            }
        )
    df = pd.DataFrame(rows)
    pred = pd.read_csv(os.path.join(args.predictions, "predictions.csv"))
    pred["row"] = np.arange(len(pred))
    df = df.merge(pred[["id", "row", "pred_model", "pred_model_median"]], on="id")
    probs = np.load(os.path.join(args.predictions, "probabilities.npy"))[df.row]
    years = np.array(json.load(open(os.path.join(args.predictions, "years.json"))))
    df["pred_expected"] = probs @ years
    df["confidence"] = probs.max(1)
    return df, probs, years


def ece(confidence, correct, bands=BANDS):
    edges = np.linspace(0, 1, bands + 1)
    band = np.clip(np.digitize(confidence, edges[1:-1]), 0, bands - 1)
    gaps = [
        (band == b).mean()
        * abs(confidence[band == b].mean() - correct[band == b].mean())
        for b in range(bands)
        if (band == b).any()
    ]
    return float(np.sum(gaps))


def year_shares(year_index, n_years, weights=None):
    """Share of documents in each year, from year indices or probabilities."""
    if weights is not None:
        return weights.mean(0)
    return np.bincount(year_index, minlength=n_years) / len(year_index)


def distances(observed, predicted):
    """Total variation distance and largest gap between cumulative shares."""
    tvd = 0.5 * np.abs(observed - predicted).sum()
    ks = np.abs(np.cumsum(observed) - np.cumsum(predicted)).max()
    return float(tvd), float(ks)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--predictions", required=True, help="folder of 9_predict_years.py")
    ap.add_argument("--dated", required=True, help="output of 3_date_html.py")
    ap.add_argument("--results", required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--seed", type=int, default=42, help="seed of the split")
    ap.add_argument("--min-year", type=int, default=1995)
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)
    res = lambda name: os.path.join(args.results, name)

    df, probs, years = load(args)
    correct = (df.pred_model == df.year).to_numpy()
    true_idx = np.searchsorted(years, df.year)
    likely_idx = np.searchsorted(years, df.pred_model)
    crawl_idx = np.searchsorted(years, df.crawl_year.clip(upper=years.max()))
    print(f"{len(df)} {args.split} documents")

    # 1. bias and accuracy per register
    rows = []
    for register in ["all"] + common.MAIN_REGISTERS:
        sel = (
            np.ones(len(df), dtype=bool)
            if register == "all"
            else common.has_register(df.registers, register).to_numpy()
        )
        if not sel.any():
            continue
        g = df[sel]
        observed = year_shares(true_idx[sel], len(years))
        dist_rows = {}
        for name, pred in [
            ("summed", year_shares(None, len(years), probs[sel])),
            ("most_likely", year_shares(likely_idx[sel], len(years))),
            ("crawl", year_shares(crawl_idx[sel], len(years))),
        ]:
            tvd, ks = distances(observed, pred)
            dist_rows[f"tvd_{name}"], dist_rows[f"ks_{name}"] = tvd, ks
        error = (g.pred_model - g.year).to_numpy()
        rows.append(
            {
                "register": register,
                "documents": int(sel.sum()),
                "exact": correct[sel].mean(),
                "bias_most_likely": (g.pred_model - g.year).mean(),
                "bias_median": (g.pred_model_median - g.year).mean(),
                "bias_expected": (g.pred_expected - g.year).mean(),
                "bias_crawl_year": (g.crawl_year - g.year).mean(),
                "ece": ece(g.confidence.to_numpy(), correct[sel]),
                "early_2plus": float((error <= -2).mean()),
                "late_2plus": float((error >= 2).mean()),
            }
            | dist_rows
        )
    by_register = pd.DataFrame(rows)

    # 2. calibration
    edges = np.linspace(0, 1, BANDS + 1)
    band = np.clip(np.digitize(df.confidence, edges[1:-1]), 0, BANDS - 1)
    reliability = pd.DataFrame(
        [
            {
                "band": f"{edges[b]:.2f}-{edges[b + 1]:.2f}",
                "documents": int((band == b).sum()),
                "confidence": df.confidence[band == b].mean(),
                "accuracy": correct[band == b].mean(),
            }
            for b in range(BANDS)
            if (band == b).any()
        ]
    )
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(df)), np.searchsorted(years, df.year)] = 1
    brier = float(((probs - onehot) ** 2).sum(1).mean())
    # ranked probability score: squared gaps between the cumulative
    # distributions, summed over years and scaled to 0-1
    k = len(years) - 1
    rps = float(((probs.cumsum(1) - onehot.cumsum(1)) ** 2).sum(1).mean() / k)
    crawl_onehot = np.zeros_like(probs)
    crawl_onehot[np.arange(len(df)), crawl_idx] = 1
    rps_crawl = float(
        ((crawl_onehot.cumsum(1) - onehot.cumsum(1)) ** 2).sum(1).mean() / k
    )
    cumulative = pd.DataFrame(
        {
            "year": years,
            "htmldate": onehot.mean(0).cumsum(),
            "summed_probabilities": probs.mean(0).cumsum(),
        }
    )

    # 3. the year distribution of the split, three ways
    dist = pd.DataFrame(
        {
            "year": years,
            "htmldate": onehot.mean(0),
            "summed_probabilities": probs.mean(0),
            "most_likely": np.bincount(
                np.searchsorted(years, df.pred_model), minlength=len(years)
            )
            / len(df),
        }
    )
    tvd = {
        m: float(0.5 * np.abs(dist[m] - dist.htmldate).sum())
        for m in ["summed_probabilities", "most_likely"]
    }
    calibration = {
        "documents": len(df),
        "ece": ece(df.confidence.to_numpy(), correct),
        "brier": brier,
        "rps": rps,
        "rps_crawl_year": rps_crawl,
        "max_cumulative_gap": float(
            (cumulative.summed_probabilities - cumulative.htmldate).abs().max()
        ),
        "tvd_year_distribution": tvd,
    }

    by_register.to_csv(res("by_register.csv"), index=False)
    reliability.to_csv(res("reliability.csv"), index=False)
    dist.to_csv(res("year_distribution.csv"), index=False)
    cumulative.to_csv(res("cumulative.csv"), index=False)
    json.dump(calibration, open(res("calibration.json"), "w"), indent=2)

    pd.set_option("display.width", 160)
    print("\nBias in years (predicted minus htmldate; > 0: too late) and accuracy:")
    print(by_register.round(3).to_string(index=False))
    print("\nReliability (confidence of the most likely year vs accuracy):")
    print(reliability.round(3).to_string(index=False))
    print(
        f"\nExpected calibration error {calibration['ece']:.3f}, "
        f"Brier score {brier:.3f}, ranked probability score {rps:.4f} "
        f"(crawl year {rps_crawl:.4f}), largest gap between the cumulative "
        f"year distributions {calibration['max_cumulative_gap']:.3f}"
    )
    print(
        "Year distribution vs htmldate, total variation distance: "
        + ", ".join(f"{m} {v:.3f}" for m, v in tvd.items())
    )
    print(f"All results in {args.results}/")


if __name__ == "__main__":
    main()
