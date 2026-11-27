"""Add the original HTML to every document in an HPLT jsonl file.

Documents from Common Crawl are fetched from data.commoncrawl.org with a
byte-range request. When HPLT's file name or offset is wrong, the page is
looked up in the crawl's URL index: first in the index files on the same data
server (through the crawl's cluster.idx, downloaded once per crawl), and only
if that fails, on the much slower index server. Other documents (Internet
Archive crawls) are fetched from the Wayback Machine. Run the same command
again to resume.

    python fetch_html.py hplt.jsonl.zst --out hplt.html.jsonl

The index and the Wayback Machine are much slower than Common Crawl's data
server. To get most documents quickly, run twice: first with --direct-only,
which leaves the slow documents out of the output, then without it, which
fetches just the documents that were left out.
"""

import argparse
import collections
import gzip
import io
import itertools
import json
import os
import re
import sys
import time
import urllib.parse

import requests
import surt
import zstandard
from warcio.archiveiterator import ArchiveIterator

CC_URL = "https://data.commoncrawl.org/"
INDEX_FILES_URL = CC_URL + "cc-index/collections/{crawl}/indexes/"
INDEX_URL = "https://index.commoncrawl.org/"
WAYBACK_URL = "https://web.archive.org/web/"
USER_AGENT = "hplt-html-fetch/0.1 (research; one request at a time)"

BASE_PAUSE = {
    "commoncrawl": 0.5,
    "index": 2.0,
    "wayback": 2.0,
}  # seconds between requests
MAX_PAUSE = 10
GIVE_UP_AFTER = 30 * 60
BROKEN_TRIES = 5  # broken responses for one page before giving up on it


def result(
    html=None, source=None, last_modified=None, wayback_timestamp=None, error=None
):
    """The fields added to each document."""
    return {
        "html": html,  # None if the page could not be fetched
        "html_source": source,  # "commoncrawl", "commoncrawl-index" or "wayback"
        "http_last_modified": last_modified,  # the website's Last-Modified header
        "wayback_timestamp": wayback_timestamp,  # the Wayback capture we actually got
        "fetch_error": error,  # why html is None
    }


def decode(content, encoding):
    try:
        return content.decode(encoding or "utf-8", errors="replace")
    except LookupError:  # an encoding name Python doesn't know
        return content.decode("utf-8", errors="replace")


def line_at_or_after(f, pos):
    """The first complete line that starts at or after byte pos, and its start."""
    if pos == 0:
        f.seek(0)
    else:
        f.seek(pos - 1)
        f.readline()  # to the end of the line that byte pos - 1 belongs to
    start = f.tell()
    return start, f.readline()


def last_line_up_to(f, key):
    """In a file of lines sorted by their first field, the last line whose first
    field is <= key (bytes), or None. A binary search over byte offsets, so the
    file is never read into memory."""
    lo, hi = 0, os.fstat(f.fileno()).st_size
    while lo < hi:  # find where the first line with a larger key starts
        mid = (lo + hi) // 2
        _, line = line_at_or_after(f, mid)
        if line and line.split(b"\t", 1)[0] <= key:
            lo = mid + 1
        else:
            hi = mid
    start, _ = line_at_or_after(f, lo)
    if start == 0:
        return None
    back = max(0, start - 65536)  # the line before it ends at start
    f.seek(back)
    return f.read(start - back).splitlines()[-1]


class Fetcher:
    def __init__(self, cache_dir, use_wayback=True, direct_only=False):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.cache_dir = cache_dir
        self.use_wayback = use_wayback
        self.direct_only = direct_only
        self.pause = dict(BASE_PAUSE)
        self.file_lists = {}
        self.cluster_indexes = {}

    def fetch(self, doc):
        """Return the fields to add to doc, or None to leave it for a later run."""
        if doc.get("crawl_id", "").startswith("CC-MAIN-"):
            return self.from_common_crawl(doc)
        if not self.use_wayback:
            return result(error="not a Common Crawl document")
        if self.direct_only:
            return None
        return self.from_wayback(doc)

    def from_common_crawl(self, doc):
        url = self.warc_url(doc)
        if url:
            page, error = self.read_record(doc, url, doc["o"], doc["s"])
            if page:
                return page
        else:
            error = "WARC file not in the crawl's file list"
        if self.direct_only:
            return None

        # Some HPLT rows point to files Common Crawl doesn't have (".warc.23.gz"),
        # or to the wrong place in a file. Then we look the page up in Common
        # Crawl's index, on the fast data server if we can.
        location = self.lookup_index_files(doc)
        if location is None:
            print(
                f"  {doc['id']}: not in the index files, asking the index server",
                file=sys.stderr,
                flush=True,
            )
            location = self.lookup_index(doc)
        if location is None:
            return result(error=f"{error}; not in the index either")
        page, error = self.read_record(doc, *location, source="commoncrawl-index")
        return page or result(error=f"{error} (location from the index)")

    def read_record(self, doc, url, start, length, source="commoncrawl"):
        """Fetch one WARC record and check that it is the page we want."""
        response, error = self.get(
            "commoncrawl",
            url,
            ok=206,
            headers={"Range": f"bytes={start}-{start + length - 1}"},
        )
        if error:
            return None, error
        try:
            record = next(ArchiveIterator(io.BytesIO(response.content)))
        except Exception:  # the bytes at that offset are not a WARC record
            return None, "no WARC record at that offset"
        if record.rec_headers.get_header("WARC-Target-URI") != doc["u"]:
            return None, "a different page at that offset"

        body = record.content_stream().read()  # undoes chunked and gzip encoding
        headers = record.http_headers
        last_modified = headers.get_header("Last-Modified") if headers else None
        return result(decode(body, doc.get("de")), source, last_modified), None

    def lookup_index_files(self, doc):
        """Find this exact capture in the crawl's index files: (url, offset, length).

        The index is sorted by URL key and timestamp, and stored in gzipped blocks
        of about 3000 lines. cluster.idx lists the first key of every block and
        where the block is, so one search in it and one small request to the data
        server find the capture.
        """
        cluster = self.cluster_index(doc["crawl_id"])
        if cluster is None:
            return None
        timestamp = re.sub(r"\D", "", doc["ts"])[:14]
        try:
            key = f"{surt.surt(doc['u'])} {timestamp}"
        except Exception:  # a URL surt cannot parse
            return None
        line = last_line_up_to(cluster, key.encode())
        if line is None:
            return None
        cdx_file, offset, length = line.decode().split("\t")[1:4]
        start, end = int(offset), int(offset) + int(length) - 1
        response, error = self.get(
            "commoncrawl",
            INDEX_FILES_URL.format(crawl=doc["crawl_id"]) + cdx_file,
            ok=206,
            headers={"Range": f"bytes={start}-{end}"},
        )
        if error:
            return None
        data = response.content
        if data.startswith(b"\x1f\x8b"):  # requests may have unzipped it
            data = gzip.decompress(data)

        # Lines look like: <url key> <timestamp> {"url": ..., "filename": ...}
        captures = [
            json.loads(cdx_line.split(" ", 2)[2])
            for cdx_line in data.decode("utf-8", errors="replace").splitlines()
            if cdx_line.startswith(key + " ")
        ]
        # Several URLs can share a key (http and https, with or without www).
        captures.sort(key=lambda capture: capture.get("url") != doc["u"])
        for capture in captures:
            if "filename" in capture:
                return (
                    CC_URL + capture["filename"],
                    int(capture["offset"]),
                    int(capture["length"]),
                )
        return None

    def cluster_index(self, crawl_id):
        """The crawl's cluster.idx as an open file, downloaded once; None if missing."""
        if crawl_id not in self.cluster_indexes:
            cached = os.path.join(self.cache_dir, crawl_id + ".cluster.idx")
            if not os.path.exists(cached):
                print(f"  downloading cluster.idx for {crawl_id}", file=sys.stderr)
                response, error = self.get(
                    "commoncrawl",
                    INDEX_FILES_URL.format(crawl=crawl_id) + "cluster.idx",
                )
                if error:
                    self.cluster_indexes[crawl_id] = None
                    return None
                with open(cached + ".tmp", "wb") as f:
                    f.write(response.content)
                os.replace(cached + ".tmp", cached)
            self.cluster_indexes[crawl_id] = open(cached, "rb")
        return self.cluster_indexes[crawl_id]

    def lookup_index(self, doc):
        """Find this exact capture with the (slow) index server: (url, offset, length)."""
        query = urllib.parse.urlencode({"url": doc["u"], "output": "json"})
        response, error = self.get(
            "index", f"{INDEX_URL}{doc['crawl_id']}-index?{query}"
        )
        if error:
            return None
        timestamp = re.sub(r"\D", "", doc["ts"])[:14]
        for line in response.text.splitlines():
            capture = json.loads(line)
            if capture.get("timestamp") == timestamp:
                return (
                    CC_URL + capture["filename"],
                    int(capture["offset"]),
                    int(capture["length"]),
                )
        return None

    def from_wayback(self, doc):
        timestamp = re.sub(r"\D", "", doc["ts"])[
            :14
        ]  # 2017-06-24T05:39:12Z -> 20170624053912
        # "id_" asks for the page exactly as archived, without the Wayback toolbar.
        response, error = self.get(
            "wayback",
            f"{WAYBACK_URL}{timestamp}id_/{doc['u']}",
            permanent=(403, 404, 410),
        )
        if error:
            return result(error="Wayback " + error)

        # We get the capture closest to the crawl time, which may not be the one HPLT used.
        captured = re.search(r"/web/(\d{14})", response.url)
        return result(
            decode(response.content, doc.get("de")),
            "wayback",
            response.headers.get("x-archive-orig-last-modified"),
            captured.group(1) if captured else None,
        )

    def warc_url(self, doc):
        """Build the WARC file's URL. HPLT's "f" field comes in three forms:

        ./segments/<segment>/warc/<file>
        <anything>/crawl-data/<crawl>/segments/<segment>/warc/<file>
        ./<file>      (no segment, so we look it up in the crawl's file list)
        """
        path = doc["f"]
        if "crawl-data/" in path:
            return CC_URL + path[path.index("crawl-data/") :]
        path = path.removeprefix("./")
        if path.startswith("segments/"):
            return f"{CC_URL}crawl-data/{doc['crawl_id']}/{path}"
        name = re.sub(
            r"\.warc\.\d+\.gz$", ".warc.gz", os.path.basename(path)
        )  # HPLT renames
        full_path = self.file_list(doc["crawl_id"]).get(name)
        return CC_URL + full_path if full_path else None

    def file_list(self, crawl_id):
        """Map file names to full paths for one crawl, from its warc.paths.gz."""
        if crawl_id not in self.file_lists:
            cached = os.path.join(self.cache_dir, crawl_id + ".warc.paths.gz")
            if not os.path.exists(cached):
                print(f"  downloading file list for {crawl_id}", file=sys.stderr)
                response, error = self.get(
                    "commoncrawl", f"{CC_URL}crawl-data/{crawl_id}/warc.paths.gz"
                )
                if error:
                    self.file_lists[crawl_id] = {}
                    return {}
                data = response.content
                if not data.startswith(b"\x1f\x8b"):  # requests may have unzipped it
                    data = gzip.compress(data)
                with open(cached + ".tmp", "wb") as f:
                    f.write(data)
                os.replace(cached + ".tmp", cached)
            with gzip.open(cached, "rt") as f:
                self.file_lists[crawl_id] = {
                    os.path.basename(p): p for p in f.read().split()
                }
        return self.file_lists[crawl_id]

    def get(self, server, url, headers=None, ok=200, permanent=(404, 416)):
        """GET politely. Returns (response, None) or (None, error message)."""
        waited = 0
        broken = 0
        for attempt in itertools.count():
            try:
                response = self.session.get(url, headers=headers, timeout=60)
            except (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ContentDecodingError,
            ) as e:
                # The response broke off midway. This tends to happen again and
                # again with the same page, so after a few tries we give up on
                # this page rather than stopping the whole run.
                response, reason = None, type(e).__name__
                broken += 1
                if broken >= BROKEN_TRIES:
                    print(
                        f"  {server}: {reason} {broken} times, giving up on this page",
                        file=sys.stderr,
                        flush=True,
                    )
                    return None, reason
            except requests.RequestException as e:
                response, reason = None, type(e).__name__
            else:
                reason = f"HTTP {response.status_code}"
                if response.status_code == ok or response.status_code in permanent:
                    break

            # Anything else (403, 429, 5xx) means "slow down". Common Crawl's data
            # server asks for retries once a second, so we do that ten times before
            # backing off exponentially, up to two minutes. The index and the
            # Wayback Machine are easily overloaded, so with them we back off at once.
            quick_tries = 10 if server == "commoncrawl" else 0
            wait = (
                1
                if attempt < quick_tries
                else min(2 ** (attempt - quick_tries + 1), 120)
            )
            if response is not None:
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    wait = max(wait, int(retry_after))
                elif response.status_code == 429:
                    wait = max(wait, 30)
            if waited >= GIVE_UP_AFTER:
                sys.exit(
                    f"{server} has refused for {waited // 60} minutes; stopping. "
                    "Run the same command later to continue."
                )
            print(
                f"  {server}: {reason}, retrying in {wait}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
            waited += wait

        # Slow down after refusals, and speed up again gradually.
        if waited:
            self.pause[server] = min(2 * self.pause[server], MAX_PAUSE)
        else:
            self.pause[server] = max(BASE_PAUSE[server], 0.9 * self.pause[server])
        time.sleep(self.pause[server])

        if response.status_code != ok:
            return None, reason
        return response, None


def read_lines(path):
    if path.endswith(".zst"):
        stream = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        return io.TextIOWrapper(stream, encoding="utf-8")
    return open(path, encoding="utf-8")


def done_ids(path):
    """Ids already in the output file. A half-written last line (Ctrl+C) is removed."""
    ids, end = set(), 0
    if os.path.exists(path):
        with open(path, "rb+") as f:
            for line in f:
                if not line.endswith(b"\n"):
                    f.truncate(end)
                    break
                ids.add(json.loads(line)["id"])
                end += len(line)
    return ids


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="HPLT .jsonl or .jsonl.zst file")
    parser.add_argument("--out", required=True, help="output .jsonl file")
    parser.add_argument("--n", type=int, help="only the first N documents")
    parser.add_argument(
        "--no-wayback",
        action="store_true",
        help="skip documents that are not from Common Crawl",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="leave out documents that need the index or the Wayback "
        "Machine; a later run without this option fetches them",
    )
    args = parser.parse_args()

    done = done_ids(args.out)
    if done:
        print(f"{len(done)} documents already done; continuing")
    cache_dir = args.out + ".paths"
    os.makedirs(cache_dir, exist_ok=True)
    fetcher = Fetcher(
        cache_dir, use_wayback=not args.no_wayback, direct_only=args.direct_only
    )

    counts = collections.Counter()
    started = time.time()
    with open(args.out, "a", encoding="utf-8") as out:
        for i, line in enumerate(read_lines(args.input)):
            if args.n is not None and i >= args.n:
                break
            doc = json.loads(line)
            if doc["id"] in done:
                continue
            done.add(doc["id"])

            try:
                fields = fetcher.fetch(doc)
            except Exception as e:  # e.g. a damaged WARC record
                print(f"  {doc['id']}: {e!r}", file=sys.stderr)
                fields = result(error=repr(e))
            if fields is None:  # left for a run without --direct-only
                counts["left for later"] += 1
            else:
                doc.update(fields)
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                out.flush()
                counts[doc["html_source"] or "failed"] += 1

            total = sum(counts.values())
            if total % 100 == 0:
                rate = total / (time.time() - started)
                print(f"{i + 1} read, {dict(counts)}, {rate:.1f} docs/s", flush=True)

    print(f"Finished: {dict(counts)}")


if __name__ == "__main__":
    main()
