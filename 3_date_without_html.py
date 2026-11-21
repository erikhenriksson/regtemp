"""Test how well htmldate dates documents without their original HTML.

Reads the output of date_html.py and runs htmldate on the fields that HPLT
documents have even when the HTML is gone:

    url        the URL alone, with an empty page
    text       the "text" field, one <p> per line
    md         the "md" field, converted to HTML
    xml        the "xml" field, with Trafilatura's tags mapped to HTML
    text+url   the "text" field together with the URL

Each condition runs in htmldate's fast and extensive mode, with the same
settings as date_html.py: original publication date, and the crawl date as
the latest possible date. The HTML-based pub_date and pub_date_extensive
are kept as the reference.

Each document becomes one small JSON line: id, URL, language, main register,
crawl date, the reference dates, and "dates" with one date (or null) per
condition, e.g. "text/fast". "snippets" holds, for the year of every date
found, the text around the first mention of that year in "text" (null if the
year is not in the text), for checking the results by hand.

    python test_text_dates.py hplt.dated.jsonl --out text_dates.jsonl --workers 4
"""

import argparse
import html
import io
import json
import multiprocessing
import re
import time
import warnings

import markdown
import zstandard
from dateutil.parser import UnknownTimezoneWarning
from htmldate import find_date

# htmldate's date parser warns about timezone names like "EDT"; harmless here.
warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)

MODES = {"fast": False, "extensive": True}
EMPTY_PAGE = "<html><body><p></p></body></html>"

# Trafilatura's XML tags and their closest HTML equivalents.
XML_TAGS = {
    "head": "h2",
    "list": "ul",
    "item": "li",
    "lb": "br",
    "hi": "b",
    "row": "tr",
    "cell": "td",
    "quote": "blockquote",
    "ref": "a",
}
XML_TAG = re.compile(r"<(/?)(%s)\b" % "|".join(XML_TAGS))


def text_to_html(text):
    paragraphs = "".join(
        f"<p>{html.escape(line)}</p>" for line in text.split("\n") if line.strip()
    )
    return f"<html><body>{paragraphs}</body></html>"


def xml_to_html(xml):
    body = XML_TAG.sub(lambda m: f"<{m[1]}{XML_TAGS[m[2]]}", xml)
    return f"<html><body>{body}</body></html>"


def md_to_html(md):
    return f"<html><body>{markdown.markdown(md)}</body></html>"


def pages(doc):
    """The page htmldate gets in each condition, and whether it gets the URL."""
    text = text_to_html(doc.get("text") or "")
    return {
        "url": (EMPTY_PAGE, True),
        "text": (text, False),
        "md": (md_to_html(doc.get("md") or ""), False),
        "xml": (xml_to_html(doc.get("xml") or ""), False),
        "text+url": (text, True),
    }


def htmldate(page, url, crawl_date, extensive):
    try:
        return find_date(
            page,
            url=url,
            original_date=True,
            extensive_search=extensive,
            max_date=crawl_date,
        )
    except Exception:  # htmldate can fail on very broken input
        return None


def snippet(text, year, width=100):
    match = re.search(rf"(?<!\d){year}(?!\d)", text)
    if not match:
        return None
    start = max(0, match.start() - width)
    return text[start : match.end() + width].replace("\n", " / ")


def main_register(doc):
    scores = {k: v for k, v in (doc.get("web-register") or {}).items() if k.isupper()}
    return max(scores, key=scores.get) if scores else None


def test_document(line):
    doc = json.loads(line)
    crawl_date = doc["ts"][:10]
    dates = {}
    for condition, (page, with_url) in pages(doc).items():
        for mode, extensive in MODES.items():
            url = doc["u"] if with_url else None
            dates[f"{condition}/{mode}"] = htmldate(page, url, crawl_date, extensive)

    found = set(dates.values()) | {doc.get("pub_date"), doc.get("pub_date_extensive")}
    years = sorted({date[:4] for date in found if date})
    text = doc.get("text") or ""
    result = {
        "id": doc.get("id"),
        "u": doc["u"],
        "lang": (doc.get("openlid_v3") or {}).get("lang", [None])[0],
        "register": main_register(doc),
        "crawl_date": crawl_date,
        "pub_date": doc.get("pub_date"),
        "pub_date_extensive": doc.get("pub_date_extensive"),
        "dates": dates,
        "snippets": {year: snippet(text, year) for year in years},
    }
    return json.dumps(result, ensure_ascii=False) + "\n"


def read_lines(path):
    if path.endswith(".zst"):
        stream = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        return io.TextIOWrapper(stream, encoding="utf-8")
    return open(path, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="output of date_html.py (.jsonl or .jsonl.zst)")
    parser.add_argument("--out", required=True, help="output .jsonl file")
    parser.add_argument("--workers", type=int, default=4, help="parallel processes")
    args = parser.parse_args()

    started = time.time()
    with (
        multiprocessing.Pool(args.workers) as pool,
        open(args.out, "w", encoding="utf-8") as out,
    ):
        lines = pool.imap(test_document, read_lines(args.input), chunksize=8)
        for i, line in enumerate(lines, 1):
            out.write(line)
            if i % 1000 == 0:
                print(
                    f"{i} documents tested, {i / (time.time() - started):.0f}/s",
                    flush=True,
                )
    print(f"Finished in {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
