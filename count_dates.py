"""Count how many documents htmldate dated, and how many can train the classifier.

Reads the output of 3_date_html.py and prints:

    how many documents got a date in fast mode (the labels), and how many
    only in extensive mode (not used)
    how many dated documents the classifier can use: a fast-mode date in
    --min-year or later (earlier dates are rare and are left out)
    how the publication dates lie before the crawl (lag in years), for the
    Results section
    how many dates fall on the crawl day itself: a high share can mean that
    pages show the current date, which htmldate then takes for the
    publication date
    documents per publication year

    python count_dates.py hplt.dated.jsonl --min-year 1995
"""

import argparse
import collections
import json


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="output of 3_date_html.py")
    parser.add_argument("--min-year", type=int, default=1995)
    args = parser.parse_args()

    total = fast = extensive_only = too_early = crawl_day = 0
    lags = collections.Counter()
    years = collections.Counter()
    for line in open(args.input, encoding="utf-8"):
        doc = json.loads(line)
        total += 1
        date = doc.get("pub_date")
        if not date:
            extensive_only += bool(doc.get("pub_date_extensive"))
            continue
        fast += 1
        year = int(date[:4])
        if year < args.min_year:
            too_early += 1
            continue
        years[year] += 1
        lag = int(doc["crawl_date"][:4]) - year
        lags[
            (
                "0"
                if lag <= 0
                else (
                    "1"
                    if lag == 1
                    else "2" if lag == 2 else "3-5" if lag <= 5 else "6+"
                )
            )
        ] += 1
        crawl_day += date == doc["crawl_date"]

    usable = fast - too_early
    share = lambda n, d: f"{100 * n / d:.1f}%" if d else "-"
    print(f"Documents:                         {total:>8}")
    print(f"Dated in fast mode (labels):       {fast:>8}  {share(fast, total)}")
    print(
        f"Dated only in extensive mode:      {extensive_only:>8}  {share(extensive_only, total)}  (not used)"
    )
    print(
        f"Not dated:                         {total - fast - extensive_only:>8}  "
        f"{share(total - fast - extensive_only, total)}"
    )
    print(f"Fast-mode dates before {args.min_year}:       {too_early:>8}  (left out)")
    print(
        f"Usable for the classifier:         {usable:>8}  {share(usable, total)} of all"
    )
    print(
        f"  dated on the crawl day itself:   {crawl_day:>8}  {share(crawl_day, usable)} of usable"
    )

    print("\nYears between publication and crawl (usable documents):")
    for name in ("0", "1", "2", "3-5", "6+"):
        print(f"  {name:<5}{lags[name]:>8}  {share(lags[name], usable)}")

    print("\nDocuments per publication year (usable documents):")
    for year in sorted(years):
        print(f"  {year}{years[year]:>8}  {share(years[year], usable)}")


if __name__ == "__main__":
    main()
