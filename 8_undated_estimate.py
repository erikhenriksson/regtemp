"""Estimate how well a trained classifier dates the documents htmldate could not date.

All test scores come from documents that htmldate dated. The others have no
label, so their accuracy cannot be measured directly. This script estimates
it in three ways, with a model trained by 6_train_classifier.py:

    1. Accuracy on test by date cue: whether the URL contains a year, whether
       the text mentions one, both, or neither.
    2. Reweighting: accuracy on test for each combination of main register and
       cue, weighted by how common each combination is among the undated
       documents. This assumes accuracy depends mainly on register and cue.
    3. Confidence: accuracy on test for each band of model confidence,
       weighted by how the undated documents fall into those bands. This
       assumes the model is as well calibrated on them as on test.

Prints the three, and writes into --results:

    by_cue.csv             test accuracy per cue, and the cue mix of test and
                           undated documents
    confidence.csv         test accuracy and the share of test and undated
                           documents per confidence band
    estimates.csv          the estimates for the undated documents
    undated_predictions.csv  predicted year, median year and confidence of
                           every undated document
    hand_check_sample.csv  a random sample of undated documents (URL, crawl
                           date, prediction, start of the text) to date by hand

    python 8_undated_estimate.py hplt.dated.jsonl --trained runs/400m_meta_start_all/best_model \\
        --train-script 6_train_classifier3.py --results runs/undated_400m_meta_start_all
"""

import argparse
import importlib.util
import json
import os
import shutil
import types

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import classifier_common as common

BANDS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


def load_all(path, seed):
    """Every document with text: the test split of the dated ones, and the undated."""
    rows = []
    for line in open(path, encoding="utf-8"):
        doc = json.loads(line)
        rows.append(
            {
                "id": doc["id"],
                "text": doc.get("text") or "",
                "url": doc["u"],
                "year": int(doc["pub_date"][:4]) if doc.get("pub_date") else np.nan,
                "crawl_year": int(doc["crawl_date"][:4]),
                "crawl_date": doc["crawl_date"][:10],
                "crawl_id": doc.get("crawl_id") or "",
                "register": common.main_register(doc.get("web-register")),
            }
        )
    df = pd.DataFrame(rows)
    df["split"] = [common.split_of(i, seed) for i in df.id]
    test = df[df.year.notna() & (df.split == "test")].reset_index(drop=True)
    undated = df[df.year.isna()].reset_index(drop=True)
    return test, undated


def cue(df, text_years):
    url = pd.Series(
        [not np.isnan(common.url_year(u, c)) for u, c in zip(df.url, df.crawl_year)]
    )
    text = pd.Series(
        [bool(text_years(t, c, 1)) for t, c in zip(df.text, df.crawl_year)]
    )
    return np.select(
        [url & text, url, text], ["URL and text", "URL only", "text only"], "neither"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("data", help="output of 3_date_html.py")
    ap.add_argument("--trained", required=True, help="best_model folder of a run")
    ap.add_argument(
        "--train-script",
        default="6_train_classifier.py",
        help="the training script, for its input encoding",
    )
    ap.add_argument("--results", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--attn", default="sdpa", help="attention implementation, as in training"
    )
    ap.add_argument("--sample", type=int, default=200, help="documents to date by hand")
    ap.add_argument(
        "--min-cell",
        type=int,
        default=20,
        help="register x cue cells with fewer test documents use the cue's accuracy",
    )
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)
    res = lambda name: os.path.join(args.results, name)

    spec = importlib.util.spec_from_file_location("train", args.train_script)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    # the input settings of the trained run
    settings = {
        "metadata": True,
        "text": "start_end",
        "max_len": 512,
        "seed": 42,
        "crawl_month": False,
        "collection": False,
        "text_years": 0,
    }
    run_config = os.path.join(
        os.path.dirname(os.path.normpath(args.trained)), "config.json"
    )
    if os.path.exists(run_config):
        saved = json.load(open(run_config))
        settings.update({k: saved[k] for k in settings if k in saved})
    else:
        print(f"no {run_config}; using default input settings {settings}")
    run = types.SimpleNamespace(**settings)

    test, undated = load_all(args.data, run.seed)
    print(f"{len(test)} test documents, {len(undated)} undated documents")
    for d in (test, undated):
        d["cue"] = cue(d, train.text_years)

    tok = AutoTokenizer.from_pretrained(args.trained)
    config = AutoConfig.from_pretrained(args.trained)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False
    years = np.array([int(config.id2label[i]) for i in range(config.num_labels)])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.trained, config=config, attn_implementation=args.attn
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=res("tmp"),
            per_device_eval_batch_size=args.batch,
            report_to="none",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        ),
        data_collator=DataCollatorWithPadding(tok),
    )
    for d in (test, undated):
        labelled = d.assign(
            year=d.year.fillna(years[0]).clip(years[0], years[-1]).astype(int)
        )
        logits = trainer.predict(train.encode(tok, labelled, years, run)).predictions
        probs = common.capped_probabilities(logits, d.crawl_year, years)
        for col, v in common.predictions(probs, years).items():
            d[col] = v

    # 1. accuracy by cue on test, and the cue mix
    by_cue = []
    for c in ["URL and text", "URL only", "text only", "neither"]:
        g = test[test.cue == c]
        by_cue.append(
            {
                "cue": c,
                "share_test": len(g) / len(test),
                "share_undated": (undated.cue == c).mean(),
            }
            | common.scores(g.year, g.pred_model)
        )
    by_cue = pd.DataFrame(by_cue)

    # 2. reweighting by register x cue
    cue_scores = {r.cue: r for r in by_cue.itertuples()}
    weights = undated.groupby(["register", "cue"]).size() / len(undated)
    estimate = {"exact": 0.0, "within_1": 0.0, "mae": 0.0}
    for (register, c), w in weights.items():
        g = test[(test.register == register) & (test.cue == c)]
        s = (
            common.scores(g.year, g.pred_model)
            if len(g) >= args.min_cell
            else cue_scores[c]._asdict()
        )
        for k in estimate:
            estimate[k] += w * s[k]

    # 3. confidence bands
    conf = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        band = lambda d: (
            d[(d.confidence > lo) & (d.confidence <= hi)]
            if lo
            else d[d.confidence <= hi]
        )
        g = band(test)
        conf.append(
            {
                "confidence": f"{lo:.1f}-{hi:.1f}",
                "share_test": len(g) / len(test),
                "share_undated": len(band(undated)) / len(undated),
            }
            | common.scores(g.year, g.pred_model)
        )
    conf = pd.DataFrame(conf)
    filled = conf.exact.notna()
    by_confidence = (
        conf.share_undated[filled] * conf.exact[filled]
    ).sum() / conf.share_undated[filled].sum()

    test_all = common.scores(test.year, test.pred_model)
    estimates = pd.DataFrame(
        [
            {
                "estimate": "measured on test (dated documents)",
                "exact": test_all["exact"],
                "within_1": test_all["within_1"],
                "mae": test_all["mae"],
            },
            {"estimate": "undated, reweighted by register x cue", **estimate},
            {
                "estimate": "undated, by confidence",
                "exact": by_confidence,
                "within_1": np.nan,
                "mae": np.nan,
            },
        ]
    )

    by_cue.to_csv(res("by_cue.csv"), index=False)
    conf.to_csv(res("confidence.csv"), index=False)
    estimates.to_csv(res("estimates.csv"), index=False)
    cols = [
        "id",
        "url",
        "crawl_date",
        "register",
        "cue",
        "pred_model",
        "pred_model_median",
        "confidence",
    ]
    undated[cols].to_csv(res("undated_predictions.csv"), index=False)
    sample = undated.sample(n=min(args.sample, len(undated)), random_state=0)
    sample.assign(text_start=sample.text.str[:500].str.replace("\n", " / "))[
        cols + ["text_start"]
    ].to_csv(res("hand_check_sample.csv"), index=False)

    pd.set_option("display.width", 160)
    print("\n1. Test accuracy by date cue, and how common each cue is:")
    print(
        by_cue[
            [
                "cue",
                "share_test",
                "share_undated",
                "n",
                "exact",
                "within_1",
                "mae",
                "bias",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\nMain registers, share of test and undated documents:")
    print(
        pd.DataFrame(
            {
                "test": test.register.value_counts(normalize=True),
                "undated": undated.register.value_counts(normalize=True),
            }
        )
        .fillna(0)
        .round(3)
        .to_string()
    )
    print("\n3. Confidence bands:")
    print(
        conf[["confidence", "share_test", "share_undated", "n", "exact", "within_1"]]
        .round(3)
        .to_string(index=False)
    )
    print("\nEstimates for the undated documents (2. and 3.):")
    print(estimates.round(3).to_string(index=False))
    print(
        f"\nPredicted crawl year itself: {(test.pred_model == test.crawl_year).mean():.1%} "
        f"of test, {(undated.pred_model == undated.crawl_year).mean():.1%} of undated"
    )
    shutil.rmtree(res("tmp"), ignore_errors=True)
    print(f"All results in {args.results}/")


if __name__ == "__main__":
    main()
