"""Logistic regression baseline for dating documents from text, URL and crawl year.

Same data, split, baselines and metrics as 6_train_classifier.py
(classifier_common.py), so the results can be compared directly.

Features: TF-IDF of words and word pairs in the beginning of the text
(--text-chars characters), TF-IDF of the words in the URL, and the crawl year
(one-hot). The model predicts the publication year, never later than the
crawl year. The regularisation strength C is chosen on dev from --c.

Writes metrics.csv, by_register.csv and predictions_<split>.csv into --results.
Test is only scored with --evaluate-test.

    python 7_logreg_baseline.py hplt.dated.jsonl --results runs/logreg
"""

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import classifier_common as common


def url_words(url):
    return " ".join(re.findall(r"[a-z]+|\d+", url.lower()))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("data", help="output of 3_date_html.py")
    ap.add_argument("--results", required=True)
    ap.add_argument("--text-chars", type=int, default=5000)
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--c", default="1,4,16", help="values of C to try on dev")
    ap.add_argument("--min-year", type=int, default=1995)
    ap.add_argument("--seed", type=int, default=42, help="seed for the split")
    ap.add_argument("--evaluate-test", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)
    res = lambda name: os.path.join(args.results, name)

    df = common.load(args.data, args.seed, args.min_year)
    parts = {
        s: df[df.split == s].reset_index(drop=True) for s in ("train", "dev", "test")
    }
    train = parts["train"]
    years = np.arange(args.min_year, df.year.max() + 1)

    text_vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_features=args.max_features, sublinear_tf=True
    )
    url_vec = TfidfVectorizer(min_df=3, sublinear_tf=True)
    crawl_enc = OneHotEncoder(handle_unknown="ignore")

    def features(d, fit=False):
        texts = [t[: args.text_chars] for t in d.text]
        urls = [url_words(u) for u in d.url]
        crawl = d[["crawl_year"]].to_numpy()
        if fit:
            return sp.hstack(
                [
                    text_vec.fit_transform(texts),
                    url_vec.fit_transform(urls),
                    crawl_enc.fit_transform(crawl),
                ]
            ).tocsr()
        return sp.hstack(
            [
                text_vec.transform(texts),
                url_vec.transform(urls),
                crawl_enc.transform(crawl),
            ]
        ).tocsr()

    x = {"train": features(train, fit=True)}
    for s in ("dev", "test"):
        x[s] = features(parts[s])
    y_train = np.searchsorted(years, train.year.to_numpy())

    def capped(model, s):
        logits = np.full((x[s].shape[0], len(years)), -np.inf)
        logits[:, model.classes_] = model.predict_log_proba(x[s])
        return common.capped_probabilities(logits, parts[s].crawl_year, years)

    best = None
    for c in [float(v) for v in args.c.split(",")]:
        started = time.time()
        model = LogisticRegression(C=c, max_iter=1000).fit(x["train"], y_train)
        probs = capped(model, "dev")
        exact = float(
            (years[probs.argmax(axis=1)] == parts["dev"].year.to_numpy()).mean()
        )
        print(
            f"C={c}: dev exact-year accuracy {exact:.3f} ({time.time() - started:.0f} s)"
        )
        if best is None or exact > best[0]:
            best = (exact, c, model)
    _, best_c, model = best
    print(f"best C on dev: {best_c}")

    methods = [
        "model",
        "model_median",
        "crawl_year",
        "gap",
        "url",
        "url_or_gap",
        "url_or_model",
    ]
    all_metrics, all_by_register = [], []
    for s in ["dev", "test"] if args.evaluate_test else ["dev"]:
        d = parts[s]
        for col, v in common.predictions(capped(model, s), years).items():
            d[col] = v
        common.add_baselines(d, train)
        d["pred_url_or_model"] = d.pred_url.fillna(d.pred_model)
        metrics, by_register = common.report(d, s, methods, set(train.website))
        all_metrics.append(metrics)
        all_by_register.append(by_register)
        d.drop(columns=["text"]).to_csv(res(f"predictions_{s}.csv"), index=False)
        print(f"\n{s} results (mae and bias in years):")
        common.print_table(metrics)

    pd.concat(all_metrics).to_csv(res("metrics.csv"), index=False)
    pd.concat(all_by_register).to_csv(res("by_register.csv"), index=False)
    with open(res("config.json"), "w") as f:
        json.dump(
            vars(args) | {"best_c": best_c, "train_documents": len(train)}, f, indent=2
        )
    print(f"\nAll results in {args.results}/")


if __name__ == "__main__":
    main()
