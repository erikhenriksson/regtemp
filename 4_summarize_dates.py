"""Summarize the output of test_text_dates.py.

The reference is pub_date, htmldate's fast mode on the original HTML. Only
documents with an HTML date count towards precision and recall, as in
htmldate's own benchmark. For each condition (e.g. "text/fast") the table
shows:

    found       share of documents with an HTML date where the condition
                found a date at all
    P/R/F1 day  precision, recall and F1 for the exact day: precision is
                correct dates / dates found, recall is correct dates /
                documents with an HTML date
    P/R/F1 yr   the same for the year
    P ±1yr      share of the dates found within one year of the HTML date
    early       share of the wrong days that are earlier than the HTML date
    jan 1       share of the dates found that are January 1, which is usually
                a bare year, such as a copyright notice
    no ref      share of documents WITHOUT an HTML date where it still found
                one: either a false positive or a date the HTML run missed

The first table covers all documents. The second covers only documents
whose URL gives no date, showing what the extracted text adds on its own.
With --by, both tables are printed per language or per main register.
With --metrics, every row of every table is also saved to a CSV.

With --disagreements, a CSV gets a random sample of the documents where a
condition disagrees with the HTML date, or found a date without one, along
with the text around the years involved, for checking by hand.

    python summarize_text_dates.py text_dates.jsonl --by lang \
        --metrics metrics.csv --disagreements check.csv
"""

import argparse
import csv
import json
import random
from collections import defaultdict


COLUMNS = ["found", "P day", "R day", "F1 day", "P yr", "R yr", "F1 yr",
           "P ±1yr", "early", "jan 1", "no ref"]


def ratio(part, whole):
    return part / whole if whole else None


def f1(precision, recall):
    if not precision or not recall:
        return None
    return 2 * precision * recall / (precision + recall)


def metrics(records, condition):
    dated = [r for r in records if r["pub_date"]]
    undated = [r for r in records if not r["pub_date"]]
    found = [(r["dates"][condition], r["pub_date"]) for r in dated if r["dates"][condition]]
    same_day = sum(date == ref for date, ref in found)
    same_year = sum(date[:4] == ref[:4] for date, ref in found)
    near_year = sum(abs(int(date[:4]) - int(ref[:4])) <= 1 for date, ref in found)
    wrong = [(date, ref) for date, ref in found if date != ref]
    p_day, r_day = ratio(same_day, len(found)), ratio(same_day, len(dated))
    p_yr, r_yr = ratio(same_year, len(found)), ratio(same_year, len(dated))
    return {
        "found": ratio(len(found), len(dated)),
        "P day": p_day, "R day": r_day, "F1 day": f1(p_day, r_day),
        "P yr": p_yr, "R yr": r_yr, "F1 yr": f1(p_yr, r_yr),
        "P ±1yr": ratio(near_year, len(found)),
        "early": ratio(sum(date < ref for date, ref in wrong), len(wrong)),
        "jan 1": ratio(sum(date[5:] == "01-01" for date, _ in found), len(found)),
        "no ref": ratio(sum(bool(r["dates"][condition]) for r in undated), len(undated)),
    }


def table(title, records, conditions, rows):
    dated = [r for r in records if r["pub_date"]]
    in_text = sum(r["snippets"].get(r["pub_date"][:4]) is not None for r in dated)
    in_text = ratio(in_text, len(dated))
    print(f"=== {title} ===")
    print(
        f"{len(dated)} documents with an HTML date, {len(records) - len(dated)} without; "
        f"year of the HTML date appears in the text: "
        f"{'-' if in_text is None else f'{in_text:.3f}'}"
    )
    print(f"{'condition':<20}" + "".join(f"{c:>8}" for c in COLUMNS))
    for condition in conditions:
        values = metrics(records, condition)
        cells = ("-" if v is None else f"{v:.3f}" for v in values.values())
        print(f"{condition:<20}" + "".join(f"{c:>8}" for c in cells))
        rows.append({"table": title, "dated": len(dated), "undated": len(records) - len(dated),
                     "condition": condition, **values})
    print()


def report(title, records, conditions, rows):
    table(f"{title}: all documents", records, conditions, rows)
    no_url = [r for r in records if not r["dates"]["url/fast"]]
    table(f"{title}: URL gives no date", no_url, conditions, rows)


def write_disagreements(records, conditions, path, per_condition, seed):
    rng = random.Random(seed)
    rows = []
    for c in conditions:
        wrong = [r for r in records
                 if r["pub_date"] and r["dates"][c] and r["dates"][c] != r["pub_date"]]
        no_ref = [r for r in records if not r["pub_date"] and r["dates"][c]]
        for kind, selected in (("wrong date", wrong), ("no HTML date", no_ref)):
            for r in rng.sample(selected, min(per_condition, len(selected))):
                found, reference = r["dates"][c], r["pub_date"]
                rows.append({
                    "condition": c,
                    "kind": kind,
                    "id": r["id"],
                    "url": r["u"],
                    "lang": r["lang"],
                    "register": r["register"],
                    "crawl_date": r["crawl_date"],
                    "html_date": reference,
                    "html_date_extensive": r["pub_date_extensive"],
                    "found_date": found,
                    "found_snippet": r["snippets"].get(found[:4]),
                    "html_snippet": r["snippets"].get(reference[:4]) if reference else None,
                })
    with open(path, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys() if rows else ["condition"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="output of test_text_dates.py")
    parser.add_argument("--by", choices=["lang", "register"], help="one report per group")
    parser.add_argument("--min-docs", type=int, default=200,
                        help="skip groups with fewer documents (with --by)")
    parser.add_argument("--metrics", help="write every table row to a CSV")
    parser.add_argument("--disagreements", help="write a CSV of disagreements to check")
    parser.add_argument("--per-condition", type=int, default=50,
                        help="CSV rows per condition and kind")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    conditions = list(records[0]["dates"])

    rows = []
    report("All", records, conditions, rows)
    if args.by:
        groups = defaultdict(list)
        for r in records:
            groups[r[args.by]].append(r)
        for name, group in sorted(groups.items(), key=lambda g: -len(g[1])):
            if len(group) >= args.min_docs:
                report(f"{args.by} = {name}", group, conditions, rows)
    if args.metrics:
        with open(args.metrics, "w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.metrics}")
    if args.disagreements:
        write_disagreements(records, conditions, args.disagreements,
                            args.per_condition, args.seed)


if __name__ == "__main__":
    main()