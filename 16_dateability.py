"""How often htmldate finds a publication date, by register and crawl year.

The classifier is trained and tested only on documents that htmldate could
date. This shows where those documents differ from the rest: for each main
register (a hybrid counts towards each; classifier_common.main_registers)
and crawl period, the number of retrieved documents and the share with a
fast-mode htmldate date.

Writes --out (one row per register and crawl year, and per register and
crawl period: documents, dated, share_dated) and prints the table by crawl
period.

    python 16_dateability.py --dated $DATA/hplt.dated.jsonl --out runs/dateability.csv
"""

import argparse
import json

import pandas as pd

import classifier_common as common

PERIODS = [("2011-2015", 2011, 2015), ("2016-2019", 2016, 2019)]
PERIODS += [("2020-2022", 2020, 2022), ("2023-2025", 2023, 2025)]


def period_of(year):
    for name, lo, hi in PERIODS:
        if lo <= year <= hi:
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dated", required=True, help="output of 3_date_html.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for line in open(args.dated, encoding="utf-8"):
        doc = json.loads(line)
        crawl_year = int(doc["crawl_date"][:4])
        registers = common.main_registers(doc.get("web-register")).split()
        for register in ["all"] + (registers or ["none"]):
            rows.append(
                {
                    "register": register,
                    "crawl_year": crawl_year,
                    "crawl_period": period_of(crawl_year),
                    "dated": bool(doc.get("pub_date")),
                }
            )
    df = pd.DataFrame(rows)
    n = int((df.register == "all").sum())
    print(f"{n} documents, {df[df.register == 'all'].dated.mean():.1%} dated")

    out = []
    for by in ["crawl_year", "crawl_period"]:
        g = df.groupby(["register", by]).dated.agg(["size", "sum", "mean"])
        g = g.reset_index().rename(
            columns={
                by: "crawl",
                "size": "documents",
                "sum": "dated",
                "mean": "share_dated",
            }
        )
        g.insert(1, "by", by)
        out.append(g)
    pd.concat(out).to_csv(args.out, index=False)

    order = ["all"] + common.MAIN_REGISTERS + ["none"]
    table = (
        df.pivot_table(index="register", columns="crawl_period", values="dated")
        .reindex([r for r in order if r in set(df.register)])
        .mul(100)
        .round(1)
    )
    table["all crawls"] = (
        df.groupby("register").dated.mean().mul(100).round(1).reindex(table.index)
    )
    table["documents"] = df.groupby("register").size().reindex(table.index)
    pd.set_option("display.width", 160)
    print("\nShare of documents with an htmldate date (%), by crawl period:")
    print(table.to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
