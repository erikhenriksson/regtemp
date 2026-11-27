"""Add publication dates from htmldate to the output of 2_select_good.py.

Each document gets three new fields:

    pub_date            htmldate's fast mode
    pub_date_extensive  htmldate's extensive mode, for comparison
    crawl_date          the date part of "ts"

Both htmldate runs look for the original publication date (not the last
update), use the document's URL, and treat the crawl date as the latest
possible date. Dates are YYYY-MM-DD, or null if htmldate found none or the
document has no HTML. All input fields are kept, in the same order.

    python 3_date_html.py hplt.good.jsonl --out hplt.dated.jsonl --workers 4
"""

import argparse
import io
import json
import multiprocessing
import time
import warnings

import zstandard
from dateutil.parser import UnknownTimezoneWarning
from htmldate import find_date

# htmldate's date parser warns about timezone names like "EDT"; harmless here.
warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)


def htmldate(doc, extensive):
    try:
        return find_date(
            doc["html"],
            url=doc["u"],
            original_date=True,
            extensive_search=extensive,
            max_date=doc["ts"][:10],
        )
    except Exception:  # htmldate can fail on very broken HTML
        return None


def date_document(line):
    doc = json.loads(line)
    if doc.get("html"):
        doc["pub_date"] = htmldate(doc, extensive=False)
        doc["pub_date_extensive"] = htmldate(doc, extensive=True)
    else:
        doc["pub_date"] = doc["pub_date_extensive"] = None
    doc["crawl_date"] = doc["ts"][:10]
    return json.dumps(doc, ensure_ascii=False) + "\n"


def read_lines(path):
    if path.endswith(".zst"):
        stream = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        return io.TextIOWrapper(stream, encoding="utf-8")
    return open(path, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="output of 2_select_good.py (.jsonl or .jsonl.zst)")
    parser.add_argument("--out", required=True, help="output .jsonl file")
    parser.add_argument("--workers", type=int, default=4, help="parallel processes")
    args = parser.parse_args()

    started = time.time()
    with (
        multiprocessing.Pool(args.workers) as pool,
        open(args.out, "w", encoding="utf-8") as out,
    ):
        lines = pool.imap(date_document, read_lines(args.input), chunksize=8)
        for i, line in enumerate(lines, 1):
            out.write(line)
            if i % 1000 == 0:
                print(
                    f"{i} documents dated, {i / (time.time() - started):.0f}/s",
                    flush=True,
                )
    print(f"Finished in {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
