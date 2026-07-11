#!/usr/bin/env python3
"""
N-LLM text-vs-label consistency scoring.  (Tiering is done by assign_tier.py, not here.)

For each item, N heterogeneous LLM judges (raters) assign a 5-point Likert fit score
(1-5) using the VERIFICATION_PROMPT below.
  - gold_label = the `gold_label` column (the source/outlet label expressed as a
                 3-class L/C/R value). We evaluate how well the inherited outlet label
                 fits the article's own title+content.
  - text       = title + content (content is truncated to MAX_CHARS; MAX_CHARS=0 means
                 no truncation).

Output (= input + dynamic columns):
  - <key>_score / <key>_reason : per-rater Likert score (1-5) and rationale
                                 (columns a, b, c, d are created for the raters used).
  - avg_score                  : mean of the valid rater scores (1-5 scale); the tier
                                 step consumes this later.
  - tier / our_label           : left blank (filled by assign_tier.py / the tier step).
  Notes:
  - If RESUME=True, scores are merged from an existing *_scored.csv by id, so raters can
    be added incrementally (e.g. b,c today -> a,d tomorrow; avg_score is recomputed over
    all judges present).
  - The row set is ALWAYS the original input (INPUT_FILES). Resume only merges scores by
    id, so even if only 5 rows were scored before, the run continues over the full input.
    Rows already scored (for the active judges) are skipped; only unscored rows/judges hit
    the API.
  - A checkpoint is written every SAVE_EVERY rows (atomic replace), so progress survives
    an interruption.

Configuration lives in the CONFIG block below (no CLI).  Run:  python3 score_consistency.py
"""
import os
import sys
import csv
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

BASE = os.path.dirname(os.path.abspath(__file__))
csv.field_size_limit(sys.maxsize)

# ===============================================================================
# CONFIG  (edit here)
# ===============================================================================
# Input CSVs to score. Relative paths are resolved against this script.
INPUT_FILES = [
    "original_dataset/abp.csv",
    # "original_dataset/checkthat.csv",
    # "original_dataset/semeval.csv",
]

# Raters (= judges). An empty model name disables that rater. provider: openai/anthropic/gemini/openrouter/mock
# judge = the curation panel (3 heterogeneous). Subject models (inference = gpt/claude) are handled elsewhere.
# Copy the exact OpenRouter slug from the openrouter.ai model page (the strings below are indicative).
RATER_A_PROVIDER, RATER_A_MODEL = "openrouter", "google/gemini-3.1-flash-lite"    # US / Western anchor
RATER_B_PROVIDER, RATER_B_MODEL = "openrouter", "qwen/qwen3.6-flash"          # CN, open-weight
RATER_C_PROVIDER, RATER_C_MODEL = "openrouter", "deepseek/deepseek-v4-flash"   # CN, open-weight
RATER_D_PROVIDER, RATER_D_MODEL = "openrouter", ""                           # unused (set a model to enable)

# API keys. Leave blank and provide them via the same-named environment variables.
# Do NOT hardcode secrets here; never commit real keys.
OPENAI_API_KEY     = ""
ANTHROPIC_API_KEY  = ""
GEMINI_API_KEY     = ""
OPENROUTER_API_KEY = ""

# Generation parameters / run options
TEMPERATURE = 0.0
MAX_TOKENS  = 700    # reasoning tokens + JSON headroom (all three judges are reasoning models)
MAX_CHARS   = 0      # content truncation length (chars). 0 = no truncation (full text); title is never truncated
WORKERS     = 200      # concurrent requests
LIMIT       = 0      # first N rows only (for testing). 0 = all
OUT_SUFFIX  = "_scored"
VERBOSE     = True   # live per-row / per-judge progress logging (cleanest at WORKERS=1)
RESUME      = True   # merge scores from an existing *_scored.csv by id (over the full input). False = start fresh
SAVE_EVERY  = 1000     # checkpoint every this many scored rows (required, since the whole input is the target)

# OpenRouter only: reproducibility / cost control
OPENROUTER_REASONING_EFFORT = "low"   # "low"/"medium"/"high" or "" (low recommended for scoring; fewer reasoning tokens)
OPENROUTER_PROVIDER = {}              # pin for reproducibility, e.g. {"order": ["DeepSeek"], "allow_fallbacks": False}
# ===============================================================================

# If a key is hardcoded above, inject it as an env var (blank = use the existing env var)
for _env, _val in [("OPENAI_API_KEY", OPENAI_API_KEY), ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
                   ("GEMINI_API_KEY", GEMINI_API_KEY), ("OPENROUTER_API_KEY", OPENROUTER_API_KEY)]:
    if _val:
        os.environ[_env] = _val


VERIFICATION_PROMPT = """[Role]
You are a media bias expert. News articles carry a source-level political leaning
label (Left / Center / Right) inherited from the outlet. Your task is to judge how
well that label is SUPPORTED by the article's own text (title + content) — i.e. the
fit between the assigned label and the evidence in the text.

IMPORTANT: You are NOT asked whether YOU agree with the label, nor to classify the
article from scratch. You are asked whether the TEXT provides evidence supporting
the assigned label "{gold_label}".

[Label definitions]
- Left: Article favors progressive viewpoints — sympathetic to social justice,
  climate action, regulation, workers' rights; critical of conservative figures/policies;
  sources lean progressive; language favors collective/governmental solutions.
- Center: Article does NOT clearly lean Left or Right. This is BROAD and includes:
  (a) Balanced reporting presenting multiple perspectives
  (b) Factual/neutral reporting without editorial voice
  (c) Mildly leaning articles whose overall lean stays within ±0.2
  (d) Articles where politics is incidental to the main topic
  Absence of strong bias is sufficient for Center.
- Right: Article favors conservative viewpoints — sympathetic to free markets,
  traditional values, deregulation, law-and-order; critical of progressive
  figures/policies; sources lean conservative; language favors individual solutions.

[How to evaluate]
Read the title and content, considering these 5 dimensions:
  1. LEXICAL TONE — loaded/emotional language vs. neutral vocabulary
  2. SOURCE SELECTION — one-sided vs. diverse quoted sources/experts
  3. FRAMING — sympathetic vs. critical portrayal of subjects
  4. OMISSION — notable absence of counter-perspectives
  5. TONE/STYLE — editorial/opinion piece vs. straight news reporting

[Fit scale — 5-point Likert]
Assign ONE integer from 1 to 5 for how well the text supports "{gold_label}":
  5 = Strongly supports. The label is clearly correct; the leaning is obvious and
      unmistakable. Any reasonable reader would immediately recognize it as
      "{gold_label}". No ambiguity.
  4 = Supports. A recognizable leaning consistent with the label, though not
      overwhelming. Most annotators would agree after reading carefully.
  3 = Mixed / neutral. Some elements support the label, others are neutral or point
      elsewhere. Reasonable annotators could disagree; an adjacent label
      (e.g., Center instead of Left) would also be defensible.
  2 = Contradicts. The content does not support the label; the text reads as a
      different leaning than the one assigned.
  1 = Strongly contradicts. The content directly contradicts the label — e.g., a
      "Left" label on strongly conservative framing, or a "Center" label on a
      clearly partisan opinion piece.

[Article title]
{title}

[Article content]
{text}

[Assigned source label]
{gold_label}

[Instructions]
1. Read the title and content, considering all 5 dimensions.
2. Starting from the assigned label "{gold_label}", assess how well the text
   supports it — do NOT classify first and then compare.
3. Assign a single integer Likert score (1-5) per the scale above.
4. Provide a brief reason citing specific textual evidence.

Output ONLY this JSON:
{{
  "score": <integer 1-5>,
  "reason": "brief explanation citing specific textual evidence (50 tokens max)"
}}"""


# ---------------------------- provider calls ----------------------------
def _post(url, headers, payload, timeout=180):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={**headers, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:                   # surface the response body (the real reason)
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {e.code}: {detail[:300]}") from None


def _chat_completion(base, key_env, model, prompt, max_tokens, token_param, temperature, extra):
    payload = {"model": model, token_param: max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        payload["temperature"] = temperature
    if extra:
        payload.update(extra)
    d = _post(f"{base}/chat/completions",
              {"Authorization": f"Bearer {os.environ[key_env]}"}, payload)
    return d["choices"][0]["message"]["content"]


def call_openai(model, prompt, temperature, max_tokens):
    # GPT-5/o-series: 'max_completion_tokens' is required; temperature must be the default (1) -> omit it
    return _chat_completion("https://api.openai.com/v1", "OPENAI_API_KEY",
                            model, prompt, max_tokens, "max_completion_tokens", None, None)


def call_openrouter(model, prompt, temperature, max_tokens):
    extra = {}
    if OPENROUTER_REASONING_EFFORT:
        extra["reasoning"] = {"effort": OPENROUTER_REASONING_EFFORT}
    if OPENROUTER_PROVIDER:
        extra["provider"] = OPENROUTER_PROVIDER
    return _chat_completion("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                            model, prompt, max_tokens, "max_tokens", temperature, extra or None)


def call_anthropic(model, prompt, temperature, max_tokens):
    # Recent Claude (sonnet-5 etc.): the temperature parameter is deprecated -> omit (use default). max_tokens required
    d = _post("https://api.anthropic.com/v1/messages",
              {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
              {"model": model, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]})
    return "".join(b.get("text", "") for b in d.get("content", []))


def call_gemini(model, prompt, temperature, max_tokens):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={os.environ['GEMINI_API_KEY']}"
    d = _post(url, {}, {"contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}})
    return d["candidates"][0]["content"]["parts"][0]["text"]


def call_mock(model, prompt, temperature, max_tokens):
    """Key-free pipeline check: deterministic score (differs per row) from a hash of the full prompt."""
    h = sum((i + 1) * ord(c) for i, c in enumerate(prompt))
    return json.dumps({"score": (h % 5) + 1, "reason": f"mock({model}) deterministic"})


PROVIDERS = {"openai": call_openai, "openrouter": call_openrouter,
             "anthropic": call_anthropic, "gemini": call_gemini, "mock": call_mock}


# ---------------------------- parsing / scoring ----------------------------
def parse_score(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in output: {text[:120]!r}")
    obj = json.loads(m.group(0))
    return max(1, min(5, int(round(float(obj["score"]))))), str(obj.get("reason", ""))[:300]


def score_one(provider, model, title, text, gold, retries=3):
    body = text if MAX_CHARS in (0, None) else text[:MAX_CHARS]   # MAX_CHARS=0 -> no content truncation
    prompt = VERIFICATION_PROMPT.format(title=title, text=body, gold_label=gold)
    last = ""
    for attempt in range(retries):
        try:
            return parse_score(PROVIDERS[provider](model, prompt, TEMPERATURE, MAX_TOKENS))
        except Exception as e:  # network / rate limit / parse failure -> backoff and retry
            last = str(e)
            if last.startswith("HTTP 4") and not last.startswith("HTTP 429"):
                break                                     # 4xx (auth/request error) won't recover -> stop now
            time.sleep(2 * (attempt + 1))
    return None, f"ERROR: {last[:200]}"


def _row_scores(row, all_keys):
    """Collect only the integer <k>_score values present on a row (prior runs + this run)."""
    out = []
    for k in all_keys:
        v = row.get(f"{k}_score")
        if v in ("", None):
            continue
        try:
            out.append(int(v))
        except (ValueError, TypeError):
            pass
    return out


def score_row(idx, row, judges, all_keys):
    gold = (row.get("gold_label") or "").capitalize()   # source label (3-class), left -> Left
    title = row.get("title") or ""
    text = row.get("content") or ""
    if VERBOSE:
        print(f"\n[row {idx}] id={row.get('id','')} gold={gold} content={len(text)} chars", flush=True)
    for key, provider, model in judges:                 # only active judges hit the API (existing columns preserved)
        t0 = time.time()
        sc, reason = score_one(provider, model, title, text, gold)
        row[f"{key}_score"] = "" if sc is None else sc
        row[f"{key}_reason"] = reason
        if VERBOSE:
            shown = f"score={sc}" if sc is not None else "score=ERR"
            clean = reason.replace("\n", " ")[:90]
            print(f"    {key} {model:26.26} {time.time()-t0:5.1f}s  {shown}  | {clean}", flush=True)
    scores = _row_scores(row, all_keys)                 # mean over all judges present (cumulative)
    row["avg_score"] = round(sum(scores) / len(scores), 1) if scores else ""
    if VERBOSE:
        print(f"    -> avg_score={row['avg_score']}  (judges={len(scores)})", flush=True)
    return row


def build_judges():
    raters = [("a", RATER_A_PROVIDER, RATER_A_MODEL), ("b", RATER_B_PROVIDER, RATER_B_MODEL),
              ("c", RATER_C_PROVIDER, RATER_C_MODEL), ("d", RATER_D_PROVIDER, RATER_D_MODEL)]
    judges = []
    for key, provider, model in raters:
        if not model:
            continue
        if provider not in PROVIDERS:
            sys.exit(f"unknown provider: {provider} (available: {', '.join(PROVIDERS)})")
        judges.append((key, provider, model))
    if not judges:
        sys.exit("no active rater -- set RATER_*_MODEL in CONFIG.")
    return judges


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, (list(rows[0].keys()) if rows else [])


def write_csv(out_path, out_cols, rows):
    """Atomic save: write to tmp then replace, so an interrupted checkpoint can't corrupt the file."""
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_path)


def process_file(path, judges):
    out_path = re.sub(r"\.csv$", OUT_SUFFIX + ".csv", path)

    # The row set is ALWAYS the original input (all rows). Resume only merges scores in.
    rows, in_cols = _read_csv(path)
    if not rows:
        print(f"[skip] {path} (empty file)"); return
    if LIMIT:
        rows = rows[:LIMIT]

    active_keys = [k for k, _, _ in judges]

    # resume: merge judge scores/avg/tier from an existing *_scored.csv into the original rows by id
    resume = RESUME and os.path.exists(out_path)
    scored_cols = []
    if resume:
        scored_rows, scored_cols = _read_csv(out_path)
        by_id = {sr.get("id"): sr for sr in scored_rows if sr.get("id") not in (None, "")}
        merge_cols = [c for c in scored_cols
                      if re.match(r"[a-d]_(score|reason)$", c) or c in ("avg_score", "tier")]
        for r in rows:
            sr = by_id.get(r.get("id"))
            if not sr:
                continue
            for c in merge_cols:
                if sr.get(c, "") != "":                 # only pull non-empty columns (do not overwrite with blanks)
                    r[c] = sr[c]

    # judge columns = those in the existing scored file U this run's active judges -> ordered a,b,c,d
    existing_keys = [k for k in "abcd" if f"{k}_score" in scored_cols]
    all_keys      = [k for k in "abcd" if k in existing_keys or k in active_keys]

    base = [c for c in in_cols
            if c not in ("avg_score", "tier") and not re.match(r"[a-d]_(score|reason)$", c)]
    judge_cols = [c for k in all_keys for c in (f"{k}_score", f"{k}_reason")]
    out_cols = base + judge_cols + ["avg_score", "tier"]
    for r in rows:
        r.setdefault("tier", "")
        for k in all_keys:                              # ensure missing judge columns exist (blank)
            r.setdefault(f"{k}_score", "")
            r.setdefault(f"{k}_reason", "")

    # to-score = rows with at least one active judge still unscored (fully scored rows are skipped)
    todo = []
    for i, r in enumerate(rows, 1):
        need = [(k, p, m) for (k, p, m) in judges if str(r.get(f"{k}_score", "")) == ""]
        if need:
            todo.append((i, r, need))

    tag = "resume/merge" if resume else "new"
    print(f"\n=== {path} ({len(rows)} rows, {tag}) -> {out_path} ===")
    print(f"    judges: active {active_keys} . after merge {all_keys} . "
          f"to score {len(todo)} rows / skipped(done) {len(rows) - len(todo)} rows")

    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(score_row, i, r, need, all_keys) for (i, r, need) in todo]
            for _ in as_completed(futs):
                done += 1
                if done % SAVE_EVERY == 0:              # checkpoint every SAVE_EVERY rows
                    write_csv(out_path, out_cols, rows)
                    print(f"  ...{done}/{len(todo)}  (checkpoint)", flush=True)

    # finalize: recompute avg_score for every row (including merged) and save
    for r in rows:
        scores = _row_scores(r, all_keys)
        r["avg_score"] = round(sum(scores) / len(scores), 1) if scores else ""
    write_csv(out_path, out_cols, rows)

    # merge .judges.json (keep prior judge records + update this run's active judges)
    jpath = re.sub(r"\.csv$", ".judges.json", out_path)
    meta = {"judges": {}}
    if resume and os.path.exists(jpath):
        try:
            meta = json.load(open(jpath)); meta.setdefault("judges", {})
        except Exception:
            meta = {"judges": {}}
    for k, p, m in judges:
        meta["judges"][k] = {"provider": p, "model": m}
    meta.update({"scale": "likert_1_5", "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                 "max_chars": MAX_CHARS, "openrouter_reasoning_effort": OPENROUTER_REASONING_EFFORT,
                 "openrouter_provider": OPENROUTER_PROVIDER})
    json.dump(meta, open(jpath, "w"), indent=2)

    valid = [r["avg_score"] for r in rows if r["avg_score"] != ""]
    print(f"  rows with avg_score: {len(valid)}/{len(rows)}"
          + (f" . mean {sum(valid)/len(valid):.1f}" if valid else ""))


def main():
    judges = build_judges()
    print(f"judges {[(k, p, m) for k, p, m in judges]} . temp {TEMPERATURE} . "
          f"max_tokens {MAX_TOKENS} . max_chars {MAX_CHARS} . workers {WORKERS}"
          + (f" . LIMIT {LIMIT}" if LIMIT else ""))
    for path in INPUT_FILES:
        full = path if os.path.isabs(path) else os.path.join(BASE, path)
        if not os.path.exists(full):
            print(f"[missing] {full}"); continue
        process_file(full, judges)
    print("\nDone. (tiers are filled by assign_tier.py)")


if __name__ == "__main__":
    main()
