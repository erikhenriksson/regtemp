"""Count how the first N documents of the sample were fetched, and which are missing.

Compares the first N documents of the sample with the output of
1_fetch_html.py and prints three lists:

    input documents by crawl     Common Crawl, or which Internet Archive crawl
    fetched, by how              commoncrawl (direct), commoncrawl-index (the
                                 offset in HPLT was wrong), wayback, or failed
                                 with its error; only the first N documents
                                 are counted, even if the output has more
    left for later               documents of the first N that are not in the
                                 output yet, and which route they need

Nothing is downloaded, and it is safe to run while 1_fetch_html.py is still
writing. The paper's counts of Common Crawl and Internet Archive documents,
of documents that needed the index, and of failed fetches come from this
script with --n 150000.

    python count_left.py samples.jsonl hplt.html.jsonl --n 150000
"""

import argparse
import collections
import io
import json
import re

import zstandard


def read_lines(path):
    if path.endswith(".zst"):
        stream = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        return io.TextIOWrapper(stream, encoding="utf-8")
    return open(path, encoding="utf-8")


def crawl_family(crawl_id):
    """CC-MAIN-2021-04 -> Common Crawl; wide00015 -> wide; archivebot -> archivebot."""
    if crawl_id.startswith("CC-MAIN-"):
        return "Common Crawl"
    return re.sub(r"\d+$", "", crawl_id) or "unknown"


def show(title, counts, total):
    print(title)
    for name, count in counts.most_common():
        print(f"  {name:<60}{count:>9}{100 * count / total:>8.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="HPLT .jsonl or .jsonl.zst file (the sample)")
    parser.add_argument("output", help="output of 1_fetch_html.py")
    parser.add_argument("--n", type=int, help="only the first N input documents")
    args = parser.parse_args()

    families = {}  # id -> crawl family, for the first N documents
    for i, line in enumerate(read_lines(args.input)):
        if args.n is not None and i >= args.n:
            break
        doc = json.loads(line)
        families[doc["id"]] = crawl_family(doc.get("crawl_id", ""))

    fetched = collections.Counter()
    done = set()
    extra = 0
    for line in read_lines(args.output):
        if not line.endswith("\n"):  # still being written by a running fetch
            break
        doc = json.loads(line)
        if doc["id"] not in families:
            extra += 1
            continue
        if doc["id"] in done:
            continue
        done.add(doc["id"])
        if doc.get("html_source"):
            fetched[doc["html_source"]] += 1
        else:
            error = re.sub(r"\d+", "N", doc.get("fetch_error") or "unknown")
            fetched["failed: " + error[:50]] += 1

    left = collections.Counter()
    for doc_id, family in families.items():
        if doc_id not in done:
            route = (
                "needs the URL index"
                if family == "Common Crawl"
                else "needs the Wayback Machine"
            )
            left[f"{family}, {route}"] += 1

    print(
        f"{len(families)} input documents read, {len(done)} of them in the output, "
        f"{sum(left.values())} left for later ({extra} other documents in the output)\n"
    )
    show(
        "Input documents by crawl:",
        collections.Counter(families.values()),
        len(families),
    )
    show("Fetched, by how (first N documents only):", fetched, max(len(done), 1))
    if left:
        show("Left for later:", left, len(families))


if __name__ == "__main__":
    main()
