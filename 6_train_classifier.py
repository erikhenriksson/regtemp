"""Train a classifier that predicts the publication year from text, URL and crawl year.

Fine-tunes an encoder (Ettin by default) on the documents that htmldate dated
in fast mode, with htmldate's year as the label. Predictions never come after
the document's crawl year.

Input. --metadata puts "crawl year | URL |" before the text. --text chooses
what of the text the model sees:

    none        no text (metadata only)
    start       the beginning of the text
    start_end   the beginning and the end, half each, when the text is too long

all within --max-len tokens. Three options add more to the metadata (each off
by default):

    --crawl-month      the crawl date as year and month ("2021-01")
    --collection       the crawl's name (a Common Crawl snapshot such as
                       CC-MAIN-2021-04, or an Internet Archive collection)
    --text-years N     the N most recent years mentioned anywhere in the text
                       (not later than the crawl year), most recent first

--label-sigma S trains on soft labels: instead of all of the target on the
correct year, a bell curve of width S years around it, so that being one year
off counts as a smaller mistake than being ten years off (0: off).

With all three, the prefix is "2021-01 | CC-MAIN-2021-04 | 2020 2019 | URL |".

Split. 80/10/10 into train, dev and test by a hash of each document id
(classifier_common.py), so every run and 7_logreg_baseline.py use the same
documents. All choices (input, model size, length) are made on dev. Test is
only scored with --evaluate-test, for the final runs.

Writes, into --results:

    config.json            the settings, training time, and inference speed
    metrics.csv            per split, subset and method: coverage, exact-year
                           accuracy, within one year, mean absolute error and
                           bias (> 0: predicted too late). Subsets: all
                           documents, and documents whose website is or is not
                           in train
    by_register.csv        the same per main register (registers by
                           classifier_common.main_registers: probability at
                           least 0.4, a hybrid counts towards each)
    predictions_<split>.csv, probabilities_<split>.npy (one row per document,
                           one column per year in years.json), best_model/

Methods: model (most likely year), model_median (median of the predicted
year distribution), crawl_year, gap (crawl year minus the median
publication-to-crawl gap in train), url (a year in the URL, where there is
one), url_or_gap, url_or_model.

    python 6_train_classifier.py hplt.dated.jsonl --model jhu-clsp/ettin-encoder-150m \\
        --metadata --text start_end --max-len 512 --results runs/150m_meta_startend_512

To score a model trained earlier, without training again (its input settings
are read from that run's config.json):

    python 6_train_classifier.py hplt.dated.jsonl --trained runs/400m_meta_start/best_model \\
        --results runs/400m_meta_start_test --evaluate-test

Running on LUMI (one GPU device per run):

    # 1. Once, on a login node: download the models into the project's cache
    export HF_HOME=/scratch/project_462001491/ehenriks/hf_cache
    python -c "from huggingface_hub import snapshot_download as d; \\
        d('jhu-clsp/ettin-encoder-150m'); d('jhu-clsp/ettin-encoder-400m')"

    # 2. An interactive GPU session with enough time (a 400M run takes ~2.5 h)
    srun --account=project_462001491 --partition=small-g --gpus-per-node=1 \\
        --cpus-per-task=7 --mem=60G --time=04:00:00 --pty bash

    # 3. In the session: check the GPU answers (should print a table at once)
    rocm-smi

    # 4. Set up (this order: the venv first, then the module) and run
    source /scratch/project_462001491/ehenriks/temporaltest/v/bin/activate
    module use /appl/local/csc/modulefiles
    module load pytorch
    export HF_HOME=/scratch/project_462001491/ehenriks/hf_cache
    export HF_HUB_OFFLINE=1
    cd /scratch/project_462001491/ehenriks/data/eng-1M
    python 6_train_classifier.py hplt.dated.jsonl --model jhu-clsp/ettin-encoder-400m \\
        --text start --results runs/400m_meta_start

Troubleshooting:

    torch.compile crashes ("A compilation subprocess exited unexpectedly"):
        the script turns torch.compile off itself (TORCH_COMPILE_DISABLE=1, and
        ModernBERT's reference_compile); check that this copy of the script is
        the one being run.
    Hangs after the document counts, and Ctrl+C does nothing: a stuck process
        still holds the GPU (amdsmi_init waits for it). Cancel the whole job
        (squeue -u $USER; scancel <id>) and start a new session, with
        --exclude=<node> for the node that hung.
    Killed at the end of training: the session's time limit; ask for more
        --time. The best checkpoint is kept in <results>/checkpoints/.
"""

import argparse
import json
import os
import re
import time

# torch.compile crashes or hangs on LUMI's AMD GPUs, and it is only a speed-up;
# this must be set before torch is imported
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

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

YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")


def text_years(text, crawl_year, n):
    """The n most recent distinct years mentioned in the text, as a string."""
    found = {int(y) for y in YEAR.findall(text) if int(y) <= crawl_year}
    return " ".join(str(y) for y in sorted(found, reverse=True)[:n])


def prefix(row, args):
    parts = [row.crawl_date[:7] if args.crawl_month else str(row.crawl_year)]
    if args.collection:
        parts.append(row.crawl_id)
    if args.text_years:
        parts.append(text_years(row.text, row.crawl_year, args.text_years) or "-")
    return " | ".join(parts + [row.url]) + " |"


def encode(tok, df, years, args):
    """Token ids and labels for each document."""
    max_len, metadata, text_part = args.max_len, args.metadata, args.text
    first = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    last = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    first, last = [t for t in [first] if t is not None], [
        t for t in [last] if t is not None
    ]
    budget = max_len - len(first) - len(last)
    chars = max_len * 8  # far more text than fits, cut before tokenizing for speed
    if text_part == "none":
        texts = [""] * len(df)
    elif text_part == "start":
        texts = [t[:chars] for t in df.text]
    else:
        texts = [
            t if len(t) <= chars else t[: chars // 2] + " … " + t[-chars // 2 :]
            for t in df.text
        ]
    prefixes = (
        [prefix(row, args) for row in df.itertuples()] if metadata else [""] * len(df)
    )
    labels = np.searchsorted(years, df.year.to_numpy())
    items = []
    for start in range(0, len(df), 1000):
        pre = tok(prefixes[start : start + 1000], add_special_tokens=False)["input_ids"]
        body = tok(texts[start : start + 1000], add_special_tokens=False)["input_ids"]
        for j, (p, ids) in enumerate(zip(pre, body)):
            p = (
                p[: budget // 4] if text_part != "none" else p[:budget]
            )  # keep the prefix short
            room = budget - len(p)
            if len(ids) > room:
                ids = (
                    ids[:room]
                    if text_part == "start"
                    else (ids[: room // 2] + ids[len(ids) - (room - room // 2) :])
                )
            input_ids = first + p + ids + last
            items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": int(labels[start + j]),
                }
            )
    return items


class SoftLabelTrainer(Trainer):
    """A Trainer whose loss spreads each label over the neighbouring years."""

    def __init__(self, *args, sigma, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits.float()
        years = torch.arange(
            logits.shape[-1], device=logits.device, dtype=torch.float32
        )
        target = torch.exp(
            -((years[None, :] - labels[:, None].float()) ** 2) / (2 * self.sigma**2)
        )
        target = target / target.sum(dim=-1, keepdim=True)
        loss = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("data", help="output of 3_date_html.py")
    ap.add_argument("--results", required=True, help="output folder")
    ap.add_argument("--model", default="jhu-clsp/ettin-encoder-150m")
    ap.add_argument(
        "--metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="put the crawl year and URL before the text (default: yes)",
    )
    ap.add_argument(
        "--text", choices=["none", "start", "start_end"], default="start_end"
    )
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument(
        "--label-sigma",
        type=float,
        default=0,
        help="soft labels: width in years of the bell curve around the correct "
        "year (0: off, train on the correct year only)",
    )
    ap.add_argument(
        "--crawl-month", action="store_true", help="crawl year and month in the prefix"
    )
    ap.add_argument(
        "--collection", action="store_true", help="the crawl's name in the prefix"
    )
    ap.add_argument(
        "--text-years",
        type=int,
        default=0,
        metavar="N",
        help="the N most recent years mentioned in the text, in the prefix",
    )
    ap.add_argument("--min-year", type=int, default=1995)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument(
        "--select",
        choices=["exact", "mae"],
        default="exact",
        help="dev metric for choosing the best epoch",
    )
    ap.add_argument("--attn", default="sdpa", help="attention implementation")
    ap.add_argument("--seed", type=int, default=42, help="seed for the split")
    ap.add_argument(
        "--train-limit", type=int, help="use only this many training documents"
    )
    ap.add_argument(
        "--trained",
        help="a best_model folder of an earlier run: skip training and only "
        "evaluate it (its input settings are taken from that run's config.json)",
    )
    ap.add_argument(
        "--evaluate-test",
        action="store_true",
        help="also score the test split (only for final runs)",
    )
    args = ap.parse_args()
    if args.trained:
        run_config = os.path.join(
            os.path.dirname(os.path.normpath(args.trained)), "config.json"
        )
        if os.path.exists(run_config):
            saved = json.load(open(run_config))
            for key in (
                "metadata",
                "text",
                "max_len",
                "min_year",
                "seed",
                "crawl_month",
                "collection",
                "text_years",
            ):
                if key in saved and saved[key] != getattr(args, key):
                    print(f"using {key}={saved[key]!r} from {run_config}")
                    setattr(args, key, saved[key])
    if not args.metadata and args.text == "none":
        ap.error("the model needs metadata, text, or both")
    if not args.metadata and (args.crawl_month or args.collection or args.text_years):
        ap.error("--crawl-month, --collection and --text-years need --metadata")
    os.makedirs(args.results, exist_ok=True)
    res = lambda name: os.path.join(args.results, name)

    df = common.load(args.data, args.seed, args.min_year)
    train = df[df.split == "train"].reset_index(drop=True)
    if args.train_limit:
        train = train.sample(n=min(args.train_limit, len(train)), random_state=0)
        train = train.reset_index(drop=True)
    splits = ["dev", "test"] if args.evaluate_test else ["dev"]
    evals = {s: df[df.split == s].reset_index(drop=True) for s in splits}
    if args.trained:
        tok = AutoTokenizer.from_pretrained(args.trained)
        config = AutoConfig.from_pretrained(args.trained)
        years = np.array([int(config.id2label[i]) for i in range(config.num_labels)])
        if df.year.min() < years[0] or df.year.max() > years[-1]:
            raise SystemExit(
                f"the data has years outside the model's {years[0]}-{years[-1]}"
            )
    else:
        years = np.arange(args.min_year, df.year.max() + 1)
        tok = AutoTokenizer.from_pretrained(args.model)
        config = AutoConfig.from_pretrained(
            args.model,
            num_labels=len(years),
            id2label={i: str(y) for i, y in enumerate(years)},
            label2id={str(y): i for i, y in enumerate(years)},
        )
    with open(res("years.json"), "w") as f:
        json.dump(years.tolist(), f)
    if hasattr(config, "reference_compile"):
        # ModernBERT (Ettin) compiles its embeddings with torch.compile by
        # default, which crashes on LUMI's AMD GPUs; it is only a speed-up
        config.reference_compile = False
    model = AutoModelForSequenceClassification.from_pretrained(
        args.trained or args.model,
        config=config,
        attn_implementation=args.attn,
        ignore_mismatched_sizes=not args.trained,  # a new year head when training
    )
    encoded = {}
    if not args.trained:
        encoded["train"] = encode(tok, train, years, args)
    for s, d in evals.items():
        encoded[s] = encode(tok, d, years, args)

    dev_crawl_years = evals["dev"].crawl_year.to_numpy()

    def dev_metrics(p):  # the same capped prediction as on test
        probs = common.capped_probabilities(p.predictions, dev_crawl_years, years)
        pred, gold = years[probs.argmax(axis=1)], years[p.label_ids]
        return {
            "exact": float((pred == gold).mean()),
            "mae": float(np.abs(pred - gold).mean()),
        }

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.trained:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=res("checkpoints"),
                per_device_eval_batch_size=args.batch * 2,
                report_to="none",
                bf16=bf16,
            ),
            data_collator=DataCollatorWithPadding(tok),
        )
        train_minutes = None
    else:
        targs = TrainingArguments(
            output_dir=res("checkpoints"),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch * 2,
            warmup_steps=int(np.ceil(len(train) / args.batch) * args.epochs) // 10,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=args.select,
            greater_is_better=args.select == "exact",
            save_total_limit=1,
            logging_steps=100,
            report_to="none",
            seed=args.seed,
            bf16=bf16,
        )
        soft = {"sigma": args.label_sigma} if args.label_sigma > 0 else {}
        trainer = (SoftLabelTrainer if soft else Trainer)(
            model=model,
            args=targs,
            **soft,
            train_dataset=encoded["train"],
            eval_dataset=encoded["dev"],
            data_collator=DataCollatorWithPadding(tok),
            compute_metrics=dev_metrics,
        )
        started = time.time()
        trainer.train()
        train_minutes = (time.time() - started) / 60
        trainer.save_model(res("best_model"))
        tok.save_pretrained(res("best_model"))

    methods = [
        "model",
        "model_median",
        "crawl_year",
        "gap",
        "url",
        "url_or_gap",
        "url_or_model",
    ]
    train_websites = set(train.website)
    gap = None
    all_metrics, all_by_register, speed = [], [], {}
    trainer.compute_metrics = None  # dev_metrics only fits the dev split
    for s, d in evals.items():
        started = time.time()
        logits = trainer.predict(encoded[s]).predictions
        speed[s] = len(d) / (time.time() - started)
        probs = common.capped_probabilities(logits, d.crawl_year, years)
        for col, v in common.predictions(probs, years).items():
            d[col] = v
        gap = common.add_baselines(d, train)
        d["pred_url_or_model"] = d.pred_url.fillna(d.pred_model)
        metrics, by_register = common.report(d, s, methods, train_websites)
        all_metrics.append(metrics)
        all_by_register.append(by_register)
        np.save(res(f"probabilities_{s}.npy"), probs.astype(np.float32))
        d.drop(columns=["text"]).to_csv(res(f"predictions_{s}.csv"), index=False)
        print(f"\n{s} results (mae and bias in years):")
        common.print_table(metrics)

    pd.concat(all_metrics).to_csv(res("metrics.csv"), index=False)
    pd.concat(all_by_register).to_csv(res("by_register.csv"), index=False)
    config = vars(args) | {
        "train_documents": len(train),
        "years": [int(years[0]), int(years[-1])],
        "median_gap_in_train": gap,
        "bf16": bf16,
        "train_minutes": round(train_minutes, 1) if train_minutes else None,
        "inference_docs_per_second": {s: round(v, 1) for s, v in speed.items()},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(res("config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(
        (f"\nTraining took {train_minutes:.0f} min; " if train_minutes else "\n")
        + "inference "
        + ", ".join(f"{s} {v:.0f} docs/s" for s, v in speed.items())
        + f". All results in {args.results}/"
    )


if __name__ == "__main__":
    main()
