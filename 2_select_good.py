"""Write the documents whose HTML is correct, in sample order.

Takes the first N documents of the sample, finds each one in the output of
1_fetch_html.py, and keeps it if its HTML was retrieved and is the page HPLT
extracted its text from. That is the same check as check_html.py: the
content looks like HTML, and at least half of up to 50 five-word runs from
HPLT's text are found in it. Prints how many documents were kept and why
the others were dropped.

    python 2_select_good.py samples.jsonl hplt.html.jsonl --n 150000 --out hplt.good.jsonl

The output file is read twice and never loaded into memory as a whole.
"""

import argparse
import collections
import html
import io
import json
import random
import re

import zstandard

HTML_START = re.compile(rb"<(html|head|body|!doctype\s+html)\b", re.I)
TAGS = re.compile(r"<[^>]*>")
WORDS = re.compile(r"\w+")
RUN = 5  # words per run
RUNS = 50  # runs looked up per document
LOW = 0.5  # documents with a lower text match are dropped


def read_lines(path):
    if path.endswith(".zst"):
        stream = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        return io.TextIOWrapper(stream, encoding="utf-8")
    return open(path, encoding="utf-8")


def looks_like_html(page):
    return bool(HTML_START.search(page[:5000].encode("utf-8", errors="replace")))


def text_match(text, page, rng):
    """Share of five-word runs from the text found in the page, or None."""
    runs = []
    for paragraph in text.split("\n"):  # runs stay within one paragraph
        words = WORDS.findall(paragraph.lower())
        runs += [" ".join(words[i : i + RUN]) for i in range(len(words) - RUN + 1)]
    if not runs:
        return None
    runs = rng.sample(runs, min(RUNS, len(runs)))
    page = page.replace("<!--", " ").replace("-->", " ")  # keep comment text
    found = (
        " " + " ".join(WORDS.findall(html.unescape(TAGS.sub(" ", page)).lower())) + " "
    )
    return sum(f" {run} " in found for run in runs) / len(runs)


def verdict(doc):
    """Why the document is dropped, or None to keep it."""
    page = doc.get("html")
    if not page:
        return "could not be retrieved"
    if not looks_like_html(page):
        start = page[:2000]
        if "mimetype" in start or "Content_Types" in start:
            return "not HTML: zipped document (Office, ODF, EPUB)"
        if start.lstrip().startswith("<"):
            return "not HTML: XML"
        return "not HTML: other"
    match = text_match(doc.get("text") or "", page, random.Random(doc["id"]))
    if match is not None and match < LOW:
        return "HTML does not match HPLT's text"
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sample", help="the HPLT sample (.jsonl or .jsonl.zst)")
    parser.add_argument("fetched", help="output of 1_fetch_html.py")
    parser.add_argument("--n", type=int, required=True, help="the first N documents")
    parser.add_argument("--out", required=True, help="output .jsonl file")
    args = parser.parse_args()

    ids = []
    for i, line in enumerate(read_lines(args.sample)):
        if i >= args.n:
            break
        ids.append(json.loads(line)["id"])
    wanted = set(ids)

    # Where each wanted document is in the fetched file.
    offsets = {}
    with open(args.fetched, "rb") as f:
        offset = 0
        for line in f:
            if not line.endswith(b"\n"):  # still being written
                break
            doc_id = json.loads(line)["id"]
            if doc_id in wanted and doc_id not in offsets:
                offsets[doc_id] = offset
            offset += len(line)

    counts = collections.Counter()
    with open(args.fetched, "rb") as f, open(args.out, "w", encoding="utf-8") as out:
        for i, doc_id in enumerate(ids, 1):
            if doc_id not in offsets:
                counts["not fetched yet"] += 1
                continue
            f.seek(offsets[doc_id])
            doc = json.loads(f.readline())
            reason = verdict(doc)
            if reason:
                counts[reason] += 1
            else:
                counts["kept"] += 1
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            if i % 10000 == 0:
                print(f"{i} documents checked", flush=True)

    print(f"\nFirst {len(ids)} documents of the sample:")
    for reason, count in counts.most_common():
        print(f"  {reason:<50}{count:>8}{100 * count / len(ids):>8.2f}%")
    print(f"\nWrote {counts['kept']} documents to {args.out}")
    if counts["not fetched yet"]:
        print("Some documents are not fetched yet: is 1_fetch_html.py still running?")


if __name__ == "__main__":
    main()
