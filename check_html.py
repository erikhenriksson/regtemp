"""Check that the HTML fetched by 1_fetch_html.py is the page HPLT extracted.

For every document with HTML, two checks:

    looks like HTML  the start of the content has an <html>, <head>, <body>
                     or <!DOCTYPE html> tag (not, say, a PDF or an image)
    text match       the share of HPLT's extracted text found in the HTML:
                     up to 50 runs of five words from "text" are looked up
                     in the HTML with its tags removed. Text inside
                     scripts, <noscript> and comments is kept, because some
                     sites put their content there (Discourse forums, for
                     example). The same page gives a share near 1; a
                     different page or version gives a low share.

Prints a table per fetch route (commoncrawl, commoncrawl-index, wayback),
and with --suspicious writes the documents with a low text match to a CSV,
with the start of their HTML, for checking by hand. Safe to run while
1_fetch_html.py is still writing.

    python check_html.py hplt.html.jsonl --suspicious suspicious.csv
"""

import argparse
import collections
import csv
import html
import json
import random
import re
import statistics

HTML_START = re.compile(rb"<(html|head|body|!doctype\s+html)\b", re.I)
TAGS = re.compile(r"<[^>]*>")
WORDS = re.compile(r"\w+")
RUN = 5  # words per run
RUNS = 50  # runs looked up per document
LOW = 0.5  # text match below this is suspicious


def looks_like_html(page):
    return bool(HTML_START.search(page[:5000].encode("utf-8", errors="replace")))


def html_words(page):
    page = page.replace("<!--", " ").replace("-->", " ")  # keep comment text
    text = TAGS.sub(" ", page)
    return " " + " ".join(WORDS.findall(html.unescape(text).lower())) + " "


def text_match(text, page, rng):
    """Share of five-word runs from the text found in the page, or None."""
    runs = []
    for paragraph in text.split("\n"):  # runs stay within one paragraph
        words = WORDS.findall(paragraph.lower())
        runs += [" ".join(words[i : i + RUN]) for i in range(len(words) - RUN + 1)]
    if not runs:
        return None
    runs = rng.sample(runs, min(RUNS, len(runs)))
    found = html_words(page)
    return sum(f" {run} " in found for run in runs) / len(runs)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="output of 1_fetch_html.py")
    parser.add_argument(
        "--suspicious", help="write documents with a low match to a CSV"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    matches = collections.defaultdict(list)
    not_html = collections.Counter()
    no_text = collections.Counter()
    suspicious = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):  # still being written
                break
            doc = json.loads(line)
            page = doc.get("html")
            if not page:
                continue
            source = doc.get("html_source") or "unknown"
            html_like = looks_like_html(page)
            not_html[source] += not html_like
            match = text_match(doc.get("text") or "", page, rng)
            if match is None:
                no_text[source] += 1
                continue
            matches[source].append(match)
            if match < LOW or not html_like:
                suspicious.append(
                    {
                        "id": doc["id"],
                        "source": source,
                        "url": doc["u"],
                        "text_match": round(match, 2),
                        "looks_like_html": html_like,
                        "html_start": re.sub(r"\s+", " ", page[:300]),
                        "text_start": re.sub(r"\s+", " ", doc["text"][:300]),
                    }
                )

    print(
        f"{'source':<20}{'docs':>8}{'not HTML':>10}{'median':>8}"
        f"{'< 0.5':>8}{'< 0.2':>8}{'no text':>9}"
    )
    for source in sorted(matches, key=lambda s: -len(matches[s])):
        values = matches[source]
        n = len(values) + no_text[source]
        print(
            f"{source:<20}{n:>8}{not_html[source] / n:>10.3f}"
            f"{statistics.median(values):>8.2f}"
            f"{sum(v < LOW for v in values) / len(values):>8.3f}"
            f"{sum(v < 0.2 for v in values) / len(values):>8.3f}"
            f"{no_text[source]:>9}"
        )
    print(
        "\nnot HTML, < 0.5, < 0.2: shares of the documents with HTML;"
        " median: median text match"
    )

    if args.suspicious:
        with open(args.suspicious, "w", encoding="utf-8", newline="") as out:
            fields = suspicious[0].keys() if suspicious else ["id"]
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(suspicious, key=lambda r: r["text_match"]))
        print(f"Wrote {len(suspicious)} suspicious documents to {args.suspicious}")


if __name__ == "__main__":
    main()
