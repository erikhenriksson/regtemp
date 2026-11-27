"""Date documents with a trained classifier, without their HTML.

Reads HPLT documents (for example samples.jsonl, which has only text, URL and
crawl time) and predicts each document's publication year with a model
trained by 6_train_classifier.py. The input is encoded exactly as in that
run (settings from its config.json), and no prediction comes after the
document's crawl year.

The first --labelled lines of the input are the documents that were dated
with htmldate and split into train, dev and test. They are marked, so that
analyses can leave them out or use only the held-out ones.

Writes, into --results:

    predictions.csv     one row per document, in input order:
                          line, id, crawl_year, labelled (first --labelled
                          lines), split (train/dev/test for labelled lines,
                          "unlabelled" otherwise), pred_model (most likely
                          year), pred_model_median (median of the predicted
                          distribution), confidence (probability of the most
                          likely year), register (main register, as in
                          classifier_common.main_register), and reg_<label>:
                          the register classifier's probability for every
                          main register and subregister (passed through, not
                          model inputs)
    probabilities.npy   the predicted distribution: one row per document (the
                        same order), one column per year in years.json
    years.json          the years of the columns
    config.json         the settings, number of documents and speed

    python 9_predict_years.py samples.jsonl --trained runs/400m_meta_start_all/best_model \\
        --results runs/predict_400m_meta_start_all

On LUMI (one GPU device; 1M documents take about 2 hours):

    sbatch --account=project_462001491 --partition=small-g --gpus-per-node=1 \\
        --cpus-per-task=7 --mem=60G --time=04:00:00 \\
        --wrap "source setup.sh && python 9_predict_years.py samples.jsonl \\
        --trained runs/400m_meta_start_all/best_model \\
        --train-script 6_train_classifier3.py --results runs/predict_400m_meta_start_all"
"""

import argparse
import importlib.util
import json
import os
import shutil
import time
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

SETTINGS = {
    "metadata": True,
    "text": "start_end",
    "max_len": 512,
    "seed": 42,
    "crawl_month": False,
    "collection": False,
    "text_years": 0,
}


def run_settings(trained):
    """The input settings of the trained run, from its config.json."""
    settings = dict(SETTINGS)
    path = os.path.join(os.path.dirname(os.path.normpath(trained)), "config.json")
    if os.path.exists(path):
        saved = json.load(open(path))
        settings.update({k: saved[k] for k in settings if k in saved})
    else:
        print(f"no {path}; using default input settings {settings}")
    return types.SimpleNamespace(**settings)


def chunks(path, size, limit):
    """Lists of up to `size` parsed documents, with their line numbers."""
    chunk = []
    for line_no, line in enumerate(open(path, encoding="utf-8")):
        if limit is not None and line_no >= limit:
            break
        chunk.append((line_no, json.loads(line)))
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def to_frame(chunk, labelled, seed):
    rows = []
    for line_no, doc in chunk:
        registers = doc.get("web-register") or {}
        rows.append(
            {
                "line": line_no,
                "id": doc["id"],
                "text": doc.get("text") or "",
                "url": doc["u"],
                "crawl_year": int(doc["ts"][:4]),
                "crawl_date": doc["ts"][:10],
                "crawl_id": doc.get("crawl_id") or "",
                "labelled": line_no < labelled,
                "split": (
                    common.split_of(doc["id"], seed)
                    if line_no < labelled
                    else "unlabelled"
                ),
                "register": common.main_register(registers),
            }
            | {f"reg_{k}": v for k, v in registers.items()}
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("data", help="HPLT documents (JSON lines with text, u, ts)")
    ap.add_argument("--trained", required=True, help="best_model folder of a run")
    ap.add_argument(
        "--train-script",
        default="6_train_classifier.py",
        help="the training script, for its input encoding",
    )
    ap.add_argument("--results", required=True)
    ap.add_argument(
        "--labelled",
        type=int,
        default=150000,
        help="the first this many lines are the htmldate-labelled set",
    )
    ap.add_argument("--n", type=int, help="only the first n documents (for tests)")
    ap.add_argument("--chunk", type=int, default=20000, help="documents per step")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--attn", default="sdpa", help="attention implementation, as in training"
    )
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)
    res = lambda name: os.path.join(args.results, name)

    spec = importlib.util.spec_from_file_location("train", args.train_script)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    run = run_settings(args.trained)

    tok = AutoTokenizer.from_pretrained(args.trained)
    config = AutoConfig.from_pretrained(args.trained)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False  # torch.compile crashes on LUMI
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

    columns, all_probs, done = None, [], 0
    started = time.time()
    for chunk in chunks(args.data, args.chunk, args.n):
        df = to_frame(chunk, args.labelled, run.seed)
        df["year"] = years[0]  # encode() needs a label; it is not used
        logits = trainer.predict(train.encode(tok, df, years, run)).predictions
        probs = common.capped_probabilities(logits, df.crawl_year, years)
        for col, v in common.predictions(probs, years).items():
            df[col] = v
        all_probs.append(probs.astype(np.float32))

        out = df.drop(columns=["text", "url", "crawl_date", "crawl_id", "year"])
        if columns is None:
            fixed = [c for c in out.columns if not c.startswith("reg_")]
            regs = sorted(c for c in out.columns if c.startswith("reg_"))
            columns = fixed + regs
        new = set(out.columns) - set(columns)
        if new:
            print(f"register labels not in the first chunk, left out: {sorted(new)}")
        out.reindex(columns=columns).to_csv(
            res("predictions.csv"),
            mode="w" if done == 0 else "a",
            header=done == 0,
            index=False,
        )
        done += len(df)
        rate = done / (time.time() - started)
        print(f"{done} documents, {rate:.1f} per second", flush=True)

    np.save(res("probabilities.npy"), np.concatenate(all_probs))
    with open(res("years.json"), "w") as f:
        json.dump(years.tolist(), f)
    with open(res("config.json"), "w") as f:
        json.dump(
            vars(args)
            | {"input_settings": vars(run), "documents": done}
            | {"docs_per_second": round(done / (time.time() - started), 1)},
            f,
            indent=2,
        )
    shutil.rmtree(res("tmp"), ignore_errors=True)
    print(f"All results in {args.results}/")


if __name__ == "__main__":
    main()
