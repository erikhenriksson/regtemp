"""Fetch HTML as fast as the servers' rules allow, and time it.

Uses 1_fetch_html.py's own fetching code, with the same output format, but
runs Common Crawl and the Wayback Machine side by side:

    Common Crawl     --cc-workers parallel workers, each pausing 0.5 s between
                     requests (8 workers: about 16 requests per second, a small
                     share of the few thousand per second Common Crawl serves)
    index server     only one request at a time, as Common Crawl's FAQ asks;
                     most lookups use the index files on the data server instead
    Wayback Machine  one worker, pausing --wayback-pause seconds between
                     requests (4 s: the reported limit of 15 per minute)

Reads documents --start to --start + --n of the sample. For a realistic timing
use documents that were not fetched before (a large --start), because Common
Crawl's HTTPS server may still have recently fetched records in its cache.

At the end, prints how fast each stream was and projects the time for
--project documents with the same mix of Common Crawl and Internet Archive
documents.

    python 0_time_fetch.py samples.jsonl --start 500000 --n 3000 --out timing.jsonl \\
        --fetch-script 1_fetch_html.py --cache-dir hplt.html.jsonl.paths
"""

import argparse
import collections
import concurrent.futures
import importlib.util
import itertools
import json
import threading
import time


def load_module(path):
    spec = importlib.util.spec_from_file_location("fetch_html", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetcher_class(module):
    """1_fetch_html.py's Fetcher, made safe to run in several threads."""
    download_lock = threading.Lock()  # one download of each cached file
    index_server_lock = threading.Lock()  # one index-server request at a time

    class Fetcher(module.Fetcher):
        def file_list(self, crawl_id):
            with download_lock:
                return super().file_list(crawl_id)

        def cluster_index(self, crawl_id):
            with download_lock:
                return super().cluster_index(crawl_id)

        def lookup_index(self, doc):
            with index_server_lock:
                return super().lookup_index(doc)

    return Fetcher


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="the HPLT sample (.jsonl or .jsonl.zst)")
    parser.add_argument("--out", required=True, help="output .jsonl file (overwritten)")
    parser.add_argument("--start", type=int, default=0, help="skip this many documents")
    parser.add_argument("--n", type=int, required=True, help="documents to fetch")
    parser.add_argument("--cc-workers", type=int, default=8)
    parser.add_argument("--wayback-pause", type=float, default=4.0)
    parser.add_argument("--fetch-script", default="1_fetch_html.py")
    parser.add_argument(
        "--cache-dir", required=True, help="1_fetch_html.py's OUT.paths folder"
    )
    parser.add_argument("--project", type=int, default=10_000_000)
    args = parser.parse_args()

    module = load_module(args.fetch_script)
    module.BASE_PAUSE["wayback"] = args.wayback_pause
    Fetcher = fetcher_class(module)

    docs = [
        json.loads(line)
        for line in itertools.islice(
            module.read_lines(args.input), args.start, args.start + args.n
        )
    ]
    common_crawl = [d for d in docs if d.get("crawl_id", "").startswith("CC-MAIN-")]
    archive = [d for d in docs if not d.get("crawl_id", "").startswith("CC-MAIN-")]
    print(
        f"{len(docs)} documents: {len(common_crawl)} Common Crawl, "
        f"{len(archive)} Internet Archive",
        flush=True,
    )

    out = open(args.out, "w", encoding="utf-8")
    write_lock = threading.Lock()
    routes = collections.Counter()
    local = threading.local()

    def fetch_one(doc):
        if not hasattr(local, "fetcher"):
            local.fetcher = Fetcher(args.cache_dir)
        try:
            fields = local.fetcher.fetch(doc)
        except BaseException as e:  # incl. 1_fetch_html.py giving up on a server
            fields = module.result(error=repr(e))
        doc.update(fields)
        with write_lock:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            routes[doc["html_source"] or "failed"] += 1
            if sum(routes.values()) % 100 == 0:
                print(f"  {dict(routes)}", flush=True)

    def run(stream, workers):
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            list(pool.map(fetch_one, stream))
        return time.monotonic() - started

    with concurrent.futures.ThreadPoolExecutor(2) as streams:
        cc = streams.submit(run, common_crawl, args.cc_workers)
        ia = streams.submit(run, archive, 1)
        cc_seconds, ia_seconds = cc.result(), ia.result()
    out.close()

    print(f"\nFetched: {dict(routes)}")
    rates = {}
    for name, n, seconds in (
        ("Common Crawl", len(common_crawl), cc_seconds),
        ("Wayback Machine", len(archive), ia_seconds),
    ):
        rates[name] = n / seconds if seconds else 0
        print(
            f"{name:<16} {n:>6} documents in {seconds / 60:6.1f} min: "
            f"{rates[name]:.2f} documents/s"
        )

    # The two streams run side by side, so the slower one sets the total time.
    days = {
        name: args.project * n / len(docs) / rates[name] / 86400
        for name, n in (
            ("Common Crawl", len(common_crawl)),
            ("Wayback Machine", len(archive)),
        )
        if rates[name]
    }
    print(f"\nProjected for {args.project:,} documents with this mix:")
    for name, d in days.items():
        print(f"  {name:<16} {d:6.1f} days")
    print(f"  {'together':<16} {max(days.values()):6.1f} days (both streams at once)")


if __name__ == "__main__":
    main()
