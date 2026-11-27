"""Draw a stratified sample of dated documents to date by hand.

Checks htmldate's labels (and the classifier) against publication dates
found by a person. The sample comes from the held-out documents (dev and
test), so the classifier's predictions for them are fair, and is stratified
by main register and publication period: --per-cell documents for each
combination of --registers and the four periods up to 2010, 2011-2015,
2016-2019 and 2020-2025 (fewer if a combination has fewer documents). A
document counts in the first combination it is drawn for. The items are
shuffled, so that registers and periods are not grouped.

Writes two files. The sheet is blind: it does not show htmldate's date or
the classifier's prediction, so that they cannot steer the annotator.

    <out>_sheet.csv   item, url, Wayback link at the crawl time, crawl date,
                      registers, start of the text, and empty columns to fill
                      in: manual_year, manual_date, date_type (published,
                      updated, copyright, other, none), evidence, register_ok
                      (yes, no, unsure), notes
    <out>_key.csv     item, document id, htmldate's date, the classifier's
                      prediction, the stratum and its size among the held-out
                      documents (for weighting in 15_score_annotation.py)

    python 14_annotation_sample.py --dated $DATA/hplt.dated.jsonl \\
        --predictions runs/predict_400m_meta_start_all --out runs/annotation
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import classifier_common as common

PERIODS = [("-2010", 1995, 2010), ("2011-2015", 2011, 2015)]
PERIODS += [("2016-2019", 2016, 2019), ("2020-2025", 2020, 2025)]


def load(path, seed, min_year):
    rows = []
    for line in open(path, encoding="utf-8"):
        doc = json.loads(line)
        if not doc.get("pub_date"):
            continue
        split = common.split_of(doc["id"], seed)
        if split == "train":
            continue
        year = max(int(doc["pub_date"][:4]), min_year)
        if year > int(doc["crawl_date"][:4]):
            continue
        rows.append(
            {
                "id": doc["id"],
                "url": doc["u"],
                "ts": doc["ts"],
                "crawl_date": doc["crawl_date"],
                "pub_date": doc["pub_date"],
                "year": year,
                "split": split,
                "html_source": doc.get("html_source", ""),
                "registers": common.main_registers(doc.get("web-register")),
                "text_start": (doc.get("text") or "")[:600].replace("\n", " / "),
            }
        )
    return pd.DataFrame(rows)


def wayback(ts, url):
    return f"https://web.archive.org/web/{''.join(ch for ch in ts if ch.isdigit())[:14]}/{url}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dated", required=True, help="output of 3_date_html.py")
    ap.add_argument("--predictions", help="folder of 9_predict_years.py")
    ap.add_argument("--out", required=True, help="path prefix of the two files")
    ap.add_argument("--registers", default="NA,IN,IP,OP,ID,HI,MT")
    ap.add_argument("--per-cell", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42, help="seed of the split")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--min-year", type=int, default=1995)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    df = load(args.dated, args.seed, args.min_year)
    if args.predictions:
        pred = pd.read_csv(os.path.join(args.predictions, "predictions.csv"))
        df = df.merge(
            pred[["id", "pred_model", "pred_model_median", "confidence"]],
            on="id",
            how="left",
        )
    rng = np.random.default_rng(args.sample_seed)
    picked, rows = set(), []
    for register in args.registers.split(","):
        has = common.has_register(df.registers, register)
        for period, lo, hi in PERIODS:
            cell = df[has & df.year.between(lo, hi)]
            free = cell[~cell.id.isin(picked)]
            n = min(args.per_cell, len(free))
            if n < args.per_cell:
                print(f"{register} {period}: only {n} documents available")
            take = free.iloc[rng.choice(len(free), n, replace=False)] if n else free
            picked.update(take.id)
            rows.append(
                take.assign(
                    stratum_register=register,
                    stratum_period=period,
                    stratum_size=len(cell),
                )
            )
    sample = pd.concat(rows).sample(frac=1, random_state=args.sample_seed)
    sample.insert(0, "item", np.arange(1, len(sample) + 1))

    sheet = sample[["item", "url", "crawl_date", "registers", "text_start"]].copy()
    sheet.insert(2, "wayback", [wayback(t, u) for t, u in zip(sample.ts, sample.url)])
    for col in [
        "manual_year",
        "manual_date",
        "date_type",
        "evidence",
        "register_ok",
        "notes",
    ]:
        sheet[col] = ""
    key_cols = ["item", "id", "split", "html_source", "pub_date", "crawl_date"]
    key_cols += [
        c for c in ["pred_model", "pred_model_median", "confidence"] if c in sample
    ]
    key_cols += ["registers", "stratum_register", "stratum_period", "stratum_size"]
    sheet.to_csv(f"{args.out}_sheet.csv", index=False, encoding="utf-8-sig")
    sample[key_cols].to_csv(f"{args.out}_key.csv", index=False)
    print(
        f"{len(sample)} documents: {args.out}_sheet.csv (to fill in), {args.out}_key.csv"
    )


if __name__ == "__main__":
    main()
