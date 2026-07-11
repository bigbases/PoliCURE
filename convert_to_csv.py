#!/usr/bin/env python3
"""
Convert the source political-bias corpora (ABP, SemEval, CheckThat) into a common
6-column CSV.

Columns: id, title, content, url, source, label, gold_label

- One CSV per corpus is written to <DATA_ROOT>/original_dataset/.
- CheckThat has no url/source in the source data, so those fields are left blank.
- `label` keeps each corpus's ORIGINAL text label; `gold_label` is that label projected
  to the common 3-class axis (left/center/right) — the field the scorer reads.

Setup: download each corpus into a directory under DATA_ROOT (this code/ directory by
default; see the README for URLs and the expected layout):
    <DATA_ROOT>/Article-Bias-Prediction/   (github.com/ramybaly/Article-Bias-Prediction)
    <DATA_ROOT>/Semeval2019t4/             (zenodo.org/records/1489920, by-publisher)
    <DATA_ROOT>/CheckThat/                 (CLEF-2023 CheckThat lab, Task 3A)

Usage:
    python3 convert_to_csv.py                 # convert all three
    python3 convert_to_csv.py abp checkthat   # convert only the given ones
"""
import os
import sys
import csv
import json
import glob
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

# -- Configuration ---------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
# Root holding the downloaded corpus directories; also receives original_dataset/.
# Defaults to this directory (the code/ repo root). Override if your data lives elsewhere.
DATA_ROOT = BASE
OUT_DIR = os.path.join(DATA_ROOT, "original_dataset")

# ABP body field: 'content_original' (raw) vs 'content' (tokenized)
ABP_CONTENT_FIELD = "content_original"
# SemEval label field: 'bias' (5-way left..right) vs 'hyperpartisan' (true/false)
SEMEVAL_LABEL_FIELD = "bias"

COLUMNS = ["id", "title", "content", "url", "source", "label", "gold_label"]
csv.field_size_limit(sys.maxsize)  # long article bodies

# Project each corpus's raw label onto the common 3-class axis (left/center/right).
# ABP and CheckThat are already L/C/R (identity); SemEval's 5-way MBFC labels collapse:
#   least -> center ; {left, left-center} -> left ; {right, right-center} -> right.
_GOLD_MAP = {
    "left": "left", "left-center": "left",
    "least": "center", "center": "center",
    "right": "right", "right-center": "right",
}


def _gold(label):
    """Project a raw label onto left/center/right; unknown values pass through lowercased."""
    v = (label or "").strip().lower()
    return _GOLD_MAP.get(v, v)


def _open_writer(name):
    """Open <OUT_DIR>/<name>.csv, write the header, and return (file, writer, path)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.csv")
    fh = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    return fh, w, path


def _domain(url):
    """Extract the outlet domain from a URL (drop a leading 'www.'); '' on failure."""
    try:
        net = urlparse(url or "").netloc.lower()
        return net[4:] if net.startswith("www.") else net
    except Exception:
        return ""


# -- ABP (AllSides) --------------------------------------------------------
def convert_abp():
    src = os.path.join(DATA_ROOT, "Article-Bias-Prediction", "data", "jsons")
    files = sorted(glob.glob(os.path.join(src, "*.json")))
    fh, w, path = _open_writer("abp")
    n = 0
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        w.writerow({
            "id": d.get("ID", ""),
            "title": d.get("title", ""),
            "content": d.get(ABP_CONTENT_FIELD, ""),
            "url": d.get("url", ""),
            "source": d.get("source", ""),
            "label": d.get("bias_text", ""),   # left / center / right
            "gold_label": _gold(d.get("bias_text", "")),
        })
        n += 1
    fh.close()
    print(f"[ABP]       {n:>7} rows -> {path}")


# -- SemEval-2019 Task 4 (by-publisher) ------------------------------------
def convert_semeval():
    art = os.path.join(DATA_ROOT, "Semeval2019t4",
                       "articles-training-bypublisher-20181122.xml")
    gt = os.path.join(DATA_ROOT, "Semeval2019t4",
                      "ground-truth-training-bypublisher-20181122.xml")

    # 1) Load ground truth (label + url) into a dict keyed by id (streaming).
    print("[SemEval]   loading ground truth...")
    labels = {}
    ctx = ET.iterparse(gt, events=("start", "end"))
    _, groot = next(ctx)
    for ev, el in ctx:
        if ev == "end" and el.tag == "article":
            labels[el.get("id")] = (el.get(SEMEVAL_LABEL_FIELD, ""), el.get("url", ""))
            el.clear(); groot.clear()
    print(f"[SemEval]   ground truth loaded ({len(labels):,}). streaming articles...")

    # 2) Stream the articles, join on id, and write.
    fh, w, path = _open_writer("semeval")
    n = 0
    ctx = ET.iterparse(art, events=("start", "end"))
    _, aroot = next(ctx)
    for ev, el in ctx:
        if ev == "end" and el.tag == "article":
            aid = el.get("id")
            label, url = labels.get(aid, ("", ""))
            content = " ".join("".join(el.itertext()).split())  # strip tags + collapse whitespace
            w.writerow({
                "id": aid,
                "title": el.get("title", ""),
                "content": content,
                "url": url,
                "source": _domain(url),         # outlet from the url domain
                "label": label,                  # bias (left..right) or hyperpartisan
                "gold_label": _gold(label),
            })
            n += 1
            if n % 100000 == 0:
                print(f"[SemEval]   ... {n:,} rows")
            el.clear(); aroot.clear()
    fh.close()
    print(f"[SemEval]   {n:>7} rows -> {path}  (label={SEMEVAL_LABEL_FIELD})")


# -- CheckThat (CLEF-2023 Task 3A) -----------------------------------------
def convert_checkthat():
    base = os.path.join(DATA_ROOT, "CheckThat", "data", "task_3A")
    fh, w, path = _open_writer("checkthat")
    n = 0
    for split in ("train_json", "dev_json"):
        for f in sorted(glob.glob(os.path.join(base, split, "*.json"))):
            if os.path.basename(f).startswith("._"):  # skip macOS metadata files
                continue
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            w.writerow({
                "id": d.get("id", ""),
                "title": d.get("title", ""),
                "content": d.get("content", ""),
                "url": "",                         # not in source -> blank
                "source": "",                      # not in source -> blank
                "label": d.get("label_text", ""),  # left / center / right
                "gold_label": _gold(d.get("label_text", "")),
            })
            n += 1
    fh.close()
    print(f"[CheckThat] {n:>7} rows -> {path}  (url/source blank)")


CONVERTERS = {
    "abp": convert_abp,
    "semeval": convert_semeval,
    "checkthat": convert_checkthat,
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(CONVERTERS)
    for t in targets:
        if t not in CONVERTERS:
            print(f"unknown dataset: {t} (available: {', '.join(CONVERTERS)})")
            continue
        CONVERTERS[t]()
    print("Done.")
