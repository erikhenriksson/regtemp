"""Compare register shares by year from two dating methods on the same documents.

Reads yearly results of 10_register_trends.py (--periods 1995..2025) for a set
of documents dated in several ways (such as --docs heldout with
--predictions) and compares each method with a reference method (htmldate),
over the nine main registers and the years from --from-year on:

    mean_diff     mean absolute difference in register share, in
                  percentage points
    max_diff      the largest absolute difference, and where it is
    in_interval   share of register-years where the method's share lies
                  inside the reference's 95% interval

    python compare_trends.py runs/years_heldout.csv --from-year 2000
"""

import argparse

import pandas as pd

import classifier_common as common


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", help="yearly output of 10_register_trends.py")
    ap.add_argument("--reference", default="htmldate")
    ap.add_argument("--from-year", type=int, default=2000)
    ap.add_argument("--to-year", type=int, default=2025)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, keep_default_na=False, dtype={"period": str})
    df = df[df.period.str.fullmatch(r"\d{4}") & df.register.isin(common.MAIN_REGISTERS)]
    for col in ["share", "low", "high"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = df.period.astype(int)
    df = df[df.year.between(args.from_year, args.to_year)]
    ref = df[df.method == args.reference].set_index(["year", "register"])

    rows = []
    for method in [m for m in df.method.unique() if m != args.reference]:
        other = df[df.method == method].set_index(["year", "register"])
        both = ref.join(other[["share"]], rsuffix="_m", how="inner").dropna(
            subset=["share", "share_m"]
        )
        diff = 100 * (both.share_m - both.share)
        worst = diff.abs().idxmax()
        inside = (both.share_m >= both.low) & (both.share_m <= both.high)
        rows.append(
            {
                "method": method,
                "mean_diff": diff.abs().mean(),
                "max_diff": diff.abs().max(),
                "max_at": f"{worst[1]} {worst[0]}",
                "in_interval": inside.mean(),
                "cells": len(both),
            }
        )
    print(
        f"{args.csv}: compared with {args.reference}, nine main registers, "
        f"{args.from_year}-{args.to_year}"
    )
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
