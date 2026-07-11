#!/usr/bin/env python3
"""
Tier assignment (final) --- measurement-anchored cutoffs + a disagreement axis.

No extra inference or re-scoring. Tiers are assigned using only the existing judge
scores (<k>_score, Likert 1-5).

Two separate axes:
  (1) Judge disagreement (response max - min >= CONTEST_RANGE) -> AMBIGUOUS
      (genuinely contested; an independent subject model is close to a coin flip here).
  (2) Consensus items are mapped by their support level:
        mean >= 4.5 (consensus level 5)        -> CLEAR      (obvious on a surface read)
        3.5 <= mean < 4.5 (consensus level 4)  -> SUBTLE     (visible only on a careful read)
        mean < 3.5 (consensus level <= 3)      -> MISMATCHED (label not supported by the text)

Why the cutoffs are not arbitrary (design rule: no hand-tuned thresholds):
  - 4.5 / 3.5 are not hand-tuned. They are the midpoints between adjacent Likert
    levels (5|4 and 4|3), i.e. a property of the measurement instrument itself.
    Cutting an ordinal scale at these level boundaries is the only principled choice,
    and it has no free parameters (so it is invariant to sample size).
  - AMBIGUOUS uses rater disagreement, the standard reliability notion. "Ambiguous"
    is not "an agreed-upon middle" but "the verdict splits": at the same mean ~= 3,
    [3,3,3] (agreement) and [2,4,5] (disagreement) are qualitatively different groups.
  - The definition never uses subject predictions, so it is non-circular. Independent
    subject accuracy is used for validation only.

Usage:
  python3 assign_tier.py --input X_scored.csv
  python3 assign_tier.py --input X_scored.csv --validate X_scored_infer.csv  # per-tier accuracy report
  options: --contest 2 (AMBIGUOUS level gap) --clear 4.5 --subtle 3.5 --output out.csv
"""
import os
import sys
import csv
import argparse
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
csv.field_size_limit(sys.maxsize)

TIERS = ["CLEAR", "SUBTLE", "AMBIGUOUS", "MISMATCHED"]   # accuracy-descending (= reliability-descending)

# -- Cutoffs (measurement-anchored). Not hand-tuned: Likert level boundaries + rater disagreement --
CLEAR_CUT   = 4.5   # consensus mean >= 4.5 (level 5)  -> CLEAR
SUBTLE_CUT  = 3.5   # consensus mean >= 3.5 (level 4)  -> SUBTLE ; below -> MISMATCHED
CONTEST_RANGE = 2   # judge response max - min >= 2 (real disagreement) -> AMBIGUOUS


def read_items(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys()) if rows else []
    jkeys = [c[:-6] for c in cols if c.endswith("_score") and c != "avg_score"]
    for r in rows:
        js = []
        for k in jkeys:
            v = r.get(f"{k}_score", "")
            if v not in ("", None):
                try:
                    js.append(int(round(float(v))))
                except ValueError:
                    pass
        r["_js"] = [x for x in js if 1 <= x <= 5]
    return rows, cols, jkeys


def tier_of(js):
    """judge score list (1-5) -> tier. Empty list -> '' (unscored)."""
    if not js:
        return ""
    if len(js) >= 2 and (max(js) - min(js)) >= CONTEST_RANGE:
        return "AMBIGUOUS"                      # rater disagreement = genuinely contested
    mean = sum(js) / len(js)                    # consensus item: map by support level
    if mean >= CLEAR_CUT:
        return "CLEAR"
    if mean >= SUBTLE_CUT:
        return "SUBTLE"
    return "MISMATCHED"


def validate(rows, infer_path):
    inf = list(csv.DictReader(open(infer_path, encoding="utf-8")))
    def norm(x): return str(x or "").strip().lower()
    gold = {r.get("id"): norm(r.get("gold_label")) for r in inf}
    subs = [c[:-5] for c in (inf[0].keys() if inf else []) if c.endswith("_pred")]
    pred = {s: {r.get("id"): norm(r.get(f"{s}_pred")) for r in inf} for s in subs}
    tier = {r.get("id"): r.get("tier", "") for r in rows}
    print(f"\n=== Validation: per-tier subject accuracy (vs gold) . subjects={subs} ===")
    print("   (tiers are defined without subjects -> this is an independent check; monotone & separated = valid)")
    for s in subs:
        cell = defaultdict(lambda: [0, 0])
        for i in gold:
            if not pred[s].get(i) or not tier.get(i):
                continue
            cell[tier[i]][1] += 1
            cell[tier[i]][0] += (pred[s][i] == gold[i])
        line = "  ".join(f"{t[:5]} {cell[t][0]/cell[t][1]:.2f}(n{cell[t][1]})"
                         for t in TIERS if cell[t][1])
        print(f"  {s}: {line}")


def main():
    global CLEAR_CUT, SUBTLE_CUT, CONTEST_RANGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="*_scored.csv (Likert, contains <k>_score)")
    ap.add_argument("--validate", help="*_infer.csv for a per-tier accuracy report")
    ap.add_argument("--contest", type=int, default=CONTEST_RANGE)
    ap.add_argument("--clear", type=float, default=CLEAR_CUT)
    ap.add_argument("--subtle", type=float, default=SUBTLE_CUT)
    ap.add_argument("--output", help="save separately (default: overwrite input)")
    args = ap.parse_args()
    CLEAR_CUT, SUBTLE_CUT, CONTEST_RANGE = args.clear, args.subtle, args.contest

    if not os.path.exists(args.input):
        sys.exit(f"[missing] {args.input}")
    rows, cols, jkeys = read_items(args.input)
    if not rows:
        sys.exit(f"[empty file] {args.input}")
    print(f"Rule: AMBIGUOUS(range>={CONTEST_RANGE}) . CLEAR(>={CLEAR_CUT}) . "
          f"SUBTLE(>={SUBTLE_CUT}) . MISMATCHED(<{SUBTLE_CUT})  . judges={jkeys}")

    dist, contested = Counter(), 0
    for r in rows:
        r["tier"] = tier_of(r["_js"])
        dist[r["tier"] or "(unscored)"] += 1

    out_cols = list(cols) + [c for c in ("tier",) if c not in cols]
    out_path = args.output or args.input
    with open(out_path, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    print(f"\n[{os.path.basename(args.input)}] {n} rows -> {out_path}")
    print("    " + " . ".join(f"{t} {dist.get(t,0)}({100*dist.get(t,0)/n:.0f}%)" for t in TIERS)
          + (f" . unscored {dist['(unscored)']}" if dist.get('(unscored)') else ""))
    if args.validate:
        validate(rows, args.validate)
    print("Done.")


if __name__ == "__main__":
    main()
