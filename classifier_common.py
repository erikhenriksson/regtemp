"""Shared code for 6_train_classifier.py and 7_logreg_baseline.py.

Both scripts load the dated documents the same way, use the same split and
the same baselines, and score predictions the same way, so their results can
be compared directly.
"""

import hashlib
import json
import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

MAIN_REGISTERS = ["MT", "LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]
# The two-level register taxonomy of HPLT's web-register labels (TurkuNLP;
# https://turkunlp.org/register-annotation-docs/abbreviations), without the
# "other" subregisters, which HPLT's register classifier does not use.
TAXONOMY = {
    "MT": [],  # machine translated or generated
    "LY": [],  # lyrical
    "SP": ["it"],  # spoken: interview
    "ID": [],  # interactive discussion
    "NA": ["ne", "sr", "nb"],  # narrative: news, sports report, narrative blog
    "HI": ["re"],  # how-to or instructions: recipe
    # informational description: encyclopedia article, research article,
    # description of a thing or person, FAQ, legal terms and conditions
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    # opinion: review, opinion blog, religious blog or sermon, advice
    "OP": ["rv", "ob", "rs", "av"],
    # informational persuasion: description with intent to sell,
    # news and opinion blog or editorial
    "IP": ["ds", "ed"],
}
ALL_REGISTERS = MAIN_REGISTERS + [s for subs in TAXONOMY.values() for s in subs]
# every label a document can get, main registers alone and with a subregister
FINE_REGISTERS = [
    label
    for main, subs in TAXONOMY.items()
    for label in [main] + [f"{main} {s}" for s in subs]
]
YEAR_IN_URL = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)")
SPLITS = {"train": 0.8, "dev": 0.1, "test": 0.1}


def website(url):
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def main_register(scores):
    """The main register with the highest score, as in 5_summarize_dates.py.

    Used for the per-register results of steps 5-8. The register analysis
    uses register_labels instead (threshold, subregisters, hybrids)."""
    scores = {k: v for k, v in (scores or {}).items() if k in MAIN_REGISTERS}
    return max(scores, key=scores.get) if scores else "none"


def register_labels(scores, threshold=0.4):
    """A document's register labels, such as ["HI re", "ID"] for a hybrid.

    Every label whose probability is at least the threshold. A subregister
    is one label together with its main register ("HI re"): a subregister
    above the threshold implies its main register, even if the main
    register's own probability is lower. A main register stands alone
    ("HI") only when none of its subregisters is above the threshold."""
    scores = scores or {}
    labels = []
    for main, subs in TAXONOMY.items():
        above = [s for s in subs if scores.get(s, 0) >= threshold]
        if above:
            labels += [f"{main} {s}" for s in above]
        elif scores.get(main, 0) >= threshold:
            labels.append(main)
    return labels


def split_of(doc_id, seed):
    """train, dev or test, from a hash of the document id: the same document
    always lands in the same split, whatever the order of the input file."""
    x = int(hashlib.md5(f"{seed}:{doc_id}".encode()).hexdigest(), 16) / 16**32
    if x < SPLITS["train"]:
        return "train"
    return "dev" if x < SPLITS["train"] + SPLITS["dev"] else "test"


def load(path, seed, min_year):
    """Documents with a fast-mode htmldate date, with their split."""
    rows, undated = [], 0
    for line in open(path, encoding="utf-8"):
        doc = json.loads(line)
        if not doc.get("pub_date"):
            undated += 1
            continue
        rows.append(
            {
                "id": doc["id"],
                "text": doc.get("text") or "",
                "url": doc["u"],
                "year": int(doc["pub_date"][:4]),
                "crawl_year": int(doc["crawl_date"][:4]),
                "crawl_date": doc["crawl_date"][:10],
                "crawl_id": doc.get("crawl_id") or "",
                "register": main_register(doc.get("web-register")),
            }
        )
    df = pd.DataFrame(rows)
    df["year"] = df.year.clip(lower=min_year)
    too_late = df.year > df.crawl_year
    df = df[~too_late].reset_index(drop=True)
    df["website"] = df.url.map(website)
    df["split"] = [split_of(i, seed) for i in df.id]
    print(
        f"{len(df)} dated documents ({undated} undated, "
        f"{int(too_late.sum())} dated after their crawl left out); "
        + ", ".join(f"{s} {int((df.split == s).sum())}" for s in SPLITS)
    )
    return df


def url_year(url, crawl_year):
    parts = urlsplit(url)
    for match in YEAR_IN_URL.findall(parts.path + "?" + parts.query):
        if int(match) <= crawl_year:
            return int(match)
    return np.nan


def capped_probabilities(logits, crawl_years, years):
    """Softmax over the years up to each document's crawl year."""
    allowed = years[None, :] <= np.asarray(crawl_years)[:, None]
    masked = np.where(allowed, logits, -np.inf)
    probs = np.exp(masked - masked.max(axis=1, keepdims=True))
    return probs / probs.sum(axis=1, keepdims=True)


def predictions(probs, years):
    """Most likely year, median year, and the probability of the most likely year."""
    return {
        "pred_model": years[probs.argmax(axis=1)],
        "pred_model_median": years[(probs.cumsum(axis=1) >= 0.5).argmax(axis=1)],
        "confidence": probs.max(axis=1),
    }


def add_baselines(df, train):
    """Baselines that need no model. df gets one pred_ column per baseline."""
    gap = int((train.crawl_year - train.year).median())
    df["pred_crawl_year"] = df.crawl_year
    df["pred_gap"] = df.crawl_year - gap
    df["pred_url"] = [url_year(u, c) for u, c in zip(df.url, df.crawl_year)]
    df["pred_url_or_gap"] = df.pred_url.fillna(df.pred_gap)
    return gap


def scores(gold, pred):
    gold, pred = np.asarray(gold, float), np.asarray(pred, float)
    has = ~np.isnan(pred)
    out = {"n": int(len(gold)), "coverage": float(has.mean()) if len(has) else np.nan}
    if not has.any():
        return out | {
            "exact": np.nan,
            "within_1": np.nan,
            "mae": np.nan,
            "bias": np.nan,
        }
    err = pred[has] - gold[has]
    return out | {
        "exact": float((err == 0).mean()),
        "within_1": float((np.abs(err) <= 1).mean()),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
    }


def report(df, split, methods, train_websites):
    """Metrics per method on all documents of a split, and separately on
    documents whose website does or does not appear in the training split."""
    seen = df.website.isin(train_websites)
    subsets = {
        "all": df,
        "website in train": df[seen],
        "website not in train": df[~seen],
    }
    rows = [
        {"split": split, "subset": name, "method": m} | scores(d.year, d[f"pred_{m}"])
        for name, d in subsets.items()
        for m in methods
    ]
    by_register = [
        {"split": split, "register": r, "method": m} | scores(g.year, g[f"pred_{m}"])
        for r, g in df.groupby("register")
        for m in methods
    ]
    print(
        f"\n{split}: {seen.mean():.1%} of documents are from a website that is also in train"
    )
    return pd.DataFrame(rows), pd.DataFrame(by_register)


def print_table(metrics):
    pd.set_option("display.width", 160)
    shown = metrics[metrics.subset == "all"].drop(columns=["subset"])
    print(shown.round(3).to_string(index=False))
