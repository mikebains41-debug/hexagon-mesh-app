"""Customer tool: turn a file of texts into a file of embeddings using Hexagon Mesh.

  python embed_file.py INPUT OUTPUT.jsonl [--field text] [--batch 5000]

INPUT can be:
  .txt    one text per line
  .jsonl  one JSON object per line; the text is in --field (default "text"); an "id" field is kept
  .csv    a column named --field (default "text"); an "id" column is kept

OUTPUT is JSON lines: {"index": 0, "id": ..., "embedding": [384 numbers]} in the same order as INPUT.
Settings: HM_URL (server address), HM_API_KEY (your key)
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("HM_URL", "http://127.0.0.1:8080").rstrip("/")
STALL_SECONDS = int(os.environ.get("HM_STALL_SECONDS", "600"))
KEY = os.environ.get("HM_API_KEY", "")


def call(method, path, data=None):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(URL + path, method=method,
                                 data=json.dumps(data).encode() if data is not None else None, headers=headers)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code >= 500 and attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            sys.exit("Server said %d: %s" % (e.code, msg))
        except urllib.error.URLError as e:
            if attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            sys.exit("Can't reach %s: %s" % (URL, e.reason))


def read_texts(path, field):
    items = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if line.strip():
                    obj = json.loads(line)
                    if not str(obj.get(field, "")).strip():
                        sys.exit("line %d has no %r text" % (n, field))
                    items.append((obj.get("id"), str(obj[field])))
    elif path.endswith(".csv"):
        with open(path, encoding="utf-8", newline="") as fh:
            for n, row in enumerate(csv.DictReader(fh), 2):
                if not str(row.get(field) or "").strip():
                    sys.exit("row %d has no %r text" % (n, field))
                items.append((row.get("id"), row[field]))
    else:
        with open(path, encoding="utf-8") as fh:
            items = [(None, line.rstrip("\n")) for line in fh if line.strip()]
    return items


def embed(items, batch=5000, replicas=None, progress=print):
    """Send texts as jobs, wait for all of them, and return (rows, job_ids, tokens) in input order."""
    jobs, tokens = [], 0
    for start in range(0, len(items), batch):
        chunk = [t for _, t in items[start:start + batch]]
        body = {"input": chunk, "model": "all-MiniLM-L6-v2"}
        if replicas:
            body["replicas"] = replicas
        r = call("POST", "/v1/jobs", body)
        jobs.append((start, r["id"]))
        tokens += r["tokens"]
        progress("  sent texts %d to %d (job %s, %s tokens)" % (start + 1, start + len(chunk), r["id"][:8],
                                                               "{:,}".format(r["tokens"])))
    done, last_count, last_change = {}, -1, time.time()
    while len(done) < len(jobs):
        for start, jid in jobs:
            if jid in done:
                continue
            r = call("GET", "/v1/jobs/" + jid)
            if r["status"] == "complete":
                done[jid] = (start, r["data"])
            elif r["status"] in ("partial", "expired"):
                sys.exit("Job %s could not be finished (%s); please run again." % (jid, r["status"]))
        finished = sum(len(v[1]) for v in done.values())
        progress("  %d of %d texts done" % (finished, len(items)))
        if finished != last_count:
            last_count, last_change = finished, time.time()
        elif time.time() - last_change > STALL_SECONDS:
            mins = max(1, STALL_SECONDS // 60)
            progress("  No progress for %d minute%s. Check the phones are plugged in, on Wi-Fi, above 80%% battery, "
                     "with the node switched on (python tools/network_status.py shows them). Checked jobs need two "
                     "phones working. The job keeps waiting; Ctrl+C stops waiting." % (mins, "" if mins == 1 else "s"))
            last_change = time.time()
        if len(done) < len(jobs):
            time.sleep(10)
    rows = []
    for start, data in sorted(done.values()):
        for d in data:
            i = start + d["index"]
            rows.append({"index": i, "id": items[i][0], "embedding": d["embedding"]})
    return rows, [jid for _, jid in jobs], tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--field", default="text")
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--single-phone", action="store_true",
                    help="operator only: one phone per ticket instead of two (ignored for customer keys)")
    a = ap.parse_args()
    items = read_texts(a.input, a.field)
    if not items:
        sys.exit("No texts found in " + a.input)
    print("%d texts from %s" % (len(items), a.input))
    rows, _, _ = embed(items, a.batch, 1 if a.single_phone else None)
    with open(a.output, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print("Wrote %d embeddings to %s" % (len(rows), a.output))


if __name__ == "__main__":
    main()
