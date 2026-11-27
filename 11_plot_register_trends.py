"""Plot register shares by publication year, one panel per main register.

Reads yearly results of 10_register_trends.py (run with --periods 1995..2025)
and draws, for each main register, its share of each year's documents with
its 95% interval as a grey band. Two kinds of plot:

    subregisters   one dating method, the main register in black and its
                   subregisters in colour (--sub: a --level sub run of the
                   same documents). A subregister line is the share of
                   documents with that subregister label ("NA ne")
    comparison     several lines, each a method of one results file, for
                   example htmldate and the classifier on the same documents.
                   The first line is the reference (black, with its interval)

Each line is given as NAME=CSV:METHOD. Years with fewer than --min-docs
documents are left out, and so is --min-year's own year (1995 by default)
before --from-year, because 10_register_trends.py clips earlier years into
it.

Subregisters that no document has (such as ed, which never reaches the
threshold) are left out, and the script says so.

Writes <register>.png and <register>.pdf into --out, all.png with every
panel, and paper.pdf, the same at the width of a two-column page.

    python 11_plot_register_trends.py --line htmldate=runs/years_dated.csv:htmldate \\
        --sub runs/years_dated_sub.csv:htmldate --out runs/plots_dated
    python 11_plot_register_trends.py --line htmldate=runs/years_heldout.csv:htmldate \\
        --line classifier=runs/years_heldout.csv:model_prob --out runs/plots_heldout
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

import classifier_common as common

# categorical colours in fixed order (validated: colour-blind safe, distinct)
COLOURS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]
GRID, TEXT = "#e5e5e5", "#555555"
DEFAULT_REGISTERS = "NA,IN,IP,OP,ID,MT"


def read(spec, from_year, min_docs):
    """Yearly rows of one method of one results file of 10_register_trends.py."""
    path, method = spec.rsplit(":", 1)
    df = pd.read_csv(path, keep_default_na=False, dtype={"period": str})  # keep NA
    df = df[(df.method == method) & df.period.str.fullmatch(r"\d{4}")].copy()
    if df.empty:
        raise SystemExit(f"{path} has no yearly rows for {method}")
    for col in ["documents", "share", "low", "high"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = df.period.astype(int)
    return df[(df.year >= from_year) & (df.documents >= min_docs)]


def end_labels(ax, ends, gap, fs):
    """Labels at the right end of the lines, moved apart so they don't overlap."""
    ends = sorted(ends, key=lambda e: e[1])
    placed = []
    for x, y, text in ends:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(
            text,
            (x, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=fs,
            color=TEXT,
        )


def panel(ax, register, lines, sub, fs=8, legend=True):
    """One register's panel; fs is the base font size. Returns the
    subregisters left out because no document has them."""
    ends = []
    single = len(lines) == 1
    for i, (name, df) in enumerate(lines):
        # the first line is the reference: black, with its interval, on top
        g = df[df.register == register].sort_values("year")
        colour = "black" if i == 0 else COLOURS[i - 1]
        label = f"{register} {common.REGISTER_NAMES[register]}" if single else name
        if i == 0:
            ax.fill_between(g.year, g.low, g.high, color="black", alpha=0.12, lw=0)
        ax.plot(g.year, g.share, color=colour, lw=2, label=label, zorder=3 - (i > 0))
        if len(g):
            ends.append(
                (g.year.iloc[-1], g.share.iloc[-1], register if single else name)
            )
    empty = []
    if sub is not None:
        for i, s in enumerate(common.TAXONOMY[register]):
            g = sub[sub.register == f"{register} {s}"].sort_values("year")
            if not (g.share > 0).any():
                empty.append(s)  # e.g. ed: never above the threshold
                continue
            ax.plot(
                g.year,
                g.share,
                color=COLOURS[i],
                lw=1.5,
                label=f"{s} {common.REGISTER_NAMES[s]}",
            )
            if len(g):
                ends.append((g.year.iloc[-1], g.share.iloc[-1], s))
    ax.set_title(
        f"{register}: {common.REGISTER_NAMES[register]}", loc="left", fontsize=fs + 2
    )
    ax.set_ylim(0, ax.get_ylim()[1] * (1.15 if legend else 1.02))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=fs, length=0)
    if legend and len(ax.get_lines()) > 1:
        box = ax.legend(fontsize=fs, loc="best", framealpha=0.9, edgecolor="none")
        box.set_zorder(5)
    if legend or single:  # comparison lines meet at the end: legend only
        end_labels(ax, ends, gap=0.045 * ax.get_ylim()[1], fs=fs)
    ax.set_xlim(right=ax.get_xlim()[1] + 1.5)  # room for the end labels
    return empty


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--line",
        action="append",
        required=True,
        metavar="NAME=CSV:METHOD",
        help="a line per register (repeat for a comparison)",
    )
    ap.add_argument(
        "--sub", metavar="CSV:METHOD", help="subregister lines (a --level sub run)"
    )
    ap.add_argument("--registers", default=DEFAULT_REGISTERS)
    ap.add_argument("--out", required=True, help="folder for the figures")
    ap.add_argument("--from-year", type=int, default=1996)
    ap.add_argument("--min-docs", type=float, default=30)
    args = ap.parse_args()
    if args.sub and len(args.line) > 1:
        ap.error("--sub goes with a single --line")
    if len(args.line) > len(COLOURS) + 1:
        ap.error(f"at most {len(COLOURS) + 1} lines")
    os.makedirs(args.out, exist_ok=True)

    lines = []
    for spec in args.line:
        name, _, rest = spec.partition("=")
        lines.append((name, read(rest, args.from_year, args.min_docs)))
    sub = read(args.sub, args.from_year, args.min_docs) if args.sub else None
    registers = args.registers.split(",")

    for register in registers:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        panel(ax, register, lines, sub)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(os.path.join(args.out, f"{register}.{ext}"), dpi=200)
        plt.close(fig)

    # all panels in one figure: all.png to look at, and paper.pdf at the
    # width of a two-column page (figure*), with smaller type and without
    # legends in the panels (subregisters are named at the end of their
    # lines; a comparison gets one legend above the panels)
    rows = int(np.ceil(len(registers) / 2))
    empty = {}
    for name, width, height, fs, legend in [
        ("all.png", 12, 3.2, 8, True),
        ("paper.pdf", 6.3, 2.1, 6, False),
    ]:
        fig, axes = plt.subplots(rows, 2, figsize=(width, height * rows), squeeze=False)
        for ax, register in zip(axes.flat, registers):
            empty[register] = panel(ax, register, lines, sub, fs=fs, legend=legend)
        for ax in axes.flat[len(registers) :]:
            ax.set_visible(False)
        shared = not legend and len(lines) > 1
        # a strip of 0.3 inches above the panels for the shared legend
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (height * rows) if shared else 1))
        if shared:
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(lines),
                fontsize=fs + 1,
                frameon=False,
            )
        fig.savefig(os.path.join(args.out, name), dpi=150)
        plt.close(fig)
    for register, subs in empty.items():
        if subs:
            print(f"{register}: no document has {', '.join(subs)}; left out")
    print(f"Wrote {len(registers)} figures, all.png and paper.pdf to {args.out}/")


if __name__ == "__main__":
    main()
