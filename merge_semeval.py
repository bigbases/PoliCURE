#!/usr/bin/env python3
"""
Reassemble the SemEval release from its split parts.

`dataset/semeval.csv` (~449 MB, 600,000 rows) is too large for GitHub's 100 MB
per-file limit, so it is committed as six ~75 MB parts under
`dataset/semeval_parts/` (semeval1.csv ... semeval6.csv). This script concatenates
them back into a single `dataset/semeval.csv` that is byte-identical to the original.

Each part is a valid CSV with the same header; the merge copies the first part
whole and appends the rest with their header line skipped.

Usage:
    python3 merge_semeval.py            # -> dataset/semeval.csv
    python3 merge_semeval.py out.csv    # -> custom output path

Standard library only; no dependencies.
"""
import os
import sys
import glob
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
PARTS_DIR = os.path.join(BASE, "dataset", "semeval_parts")
DEFAULT_OUT = os.path.join(BASE, "dataset", "semeval.csv")


def _part_index(path):
    """Sort key: the integer in a filename like 'semeval3.csv' -> 3."""
    digits = "".join(ch for ch in os.path.basename(path) if ch.isdigit())
    return int(digits) if digits else 0


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "semeval*.csv")), key=_part_index)
    if not parts:
        sys.exit(f"No parts found in {PARTS_DIR} (expected semeval1.csv ... semeval6.csv)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total = 0
    with open(out_path, "wb") as out:
        for i, part in enumerate(parts):
            with open(part, "rb") as f:
                if i > 0:
                    f.readline()          # drop the repeated header row
                shutil.copyfileobj(f, out)
            total += os.path.getsize(part)
            print(f"  + {os.path.basename(part)}")
    print(f"Merged {len(parts)} parts -> {out_path} ({os.path.getsize(out_path):,} bytes)")


if __name__ == "__main__":
    main()
