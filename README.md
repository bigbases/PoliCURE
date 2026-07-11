# PoliCURE — Code and Data

PoliCURE is a difficulty-tiered political-bias benchmark. Rather than producing new
labels, it measures how strongly each article's text supports the **source-level
(outlet-inherited)** label it already carries, and stratifies items into four
reliability tiers. This repository contains the **ID-only public release** of the
tiered data and the **scripts** that reproduce the pipeline end to end.

The pipeline is three dependency-free steps (Python 3.8+, standard library only):

0. **`convert_to_csv.py`** — download the three source corpora (ABP, SemEval,
   CheckThat) and unify them into a common `id,title,content,url,source,label,gold_label`
   CSV, where `gold_label` is the source label projected to `left`/`center`/`right`.
   See *Getting the source corpora*.
1. **`score_consistency.py`** — a panel of *N* heterogeneous LLM judges reads each
   article and rates, on a 1–5 Likert scale, how well the text supports the assigned
   source label. Judges act as *evidence evaluators*, not classifiers.
2. **`assign_tier.py`** — maps the panel scores to a tier with a fixed,
   free-parameter rule (no re-scoring, no model training).

---

## Repository layout

```
code/
├── README.md                 # this file
├── LICENSE                   # MIT license for the scripts
├── LICENSE-DATA              # CC BY 4.0 license for dataset/ artifacts
├── convert_to_csv.py         # stage 0: source corpora -> unified full-text CSV
├── score_consistency.py      # stage 1: LLM-judge consistency scoring
├── assign_tier.py            # stage 2: score -> tier assignment
├── merge_semeval.py          # reassemble dataset/semeval.csv from its split parts
└── dataset/                  # the ID-only public release
    ├── abp.csv               # ABP        (37,554 items)
    ├── checkthat.csv         # CheckThat  (50,074 items)
    ├── semeval.csv           # SemEval    (600,000 items) — rebuild via merge_semeval.py (git-ignored)
    ├── semeval_parts/        # semeval1.csv ... semeval6.csv (~75 MB each; committed)
    ├── abp.judges.json       # judge panel + settings used to score ABP
    ├── checkthat.judges.json
    └── semeval.judges.json
```

The **full-text intermediate files** (`*_scored.csv`, which keep the article
`title`/`content`) are **not** included in this repository, both to respect the
publishers' copyright and because of their size. They live outside `code/` and are
reconstructed by joining the release files back to the original corpora (below).

---

## Getting the source corpora

The release is ID-only. To (re)build the full-text CSVs, download the three source
corpora and place each in a directory inside **`code/`** (the repo root, which is
`DATA_ROOT` in `convert_to_csv.py` and is already `.gitignore`d), then run
`convert_to_csv.py`.

| corpus | download | place in |
|--------|----------|----------|
| ABP (AllSides) | `git clone https://github.com/ramybaly/Article-Bias-Prediction.git` | `Article-Bias-Prediction/` |
| CheckThat! 2023 — Task 3A | `git clone https://gitlab.com/checkthat_lab/clef2023-checkthat-lab.git` (use Task 3A) | `CheckThat/` |
| SemEval-2019 Task 4 (by-publisher) | [zenodo.org/records/1489920](https://zenodo.org/records/1489920) — download and unzip `articles-training-bypublisher-20181122.zip` and `ground-truth-training-bypublisher-20181122.zip` | `Semeval2019t4/` |

Expected layout under `code/`:

```
Article-Bias-Prediction/data/jsons/*.json
CheckThat/data/task_3A/{train_json,dev_json}/*.json
Semeval2019t4/articles-training-bypublisher-20181122.xml
Semeval2019t4/ground-truth-training-bypublisher-20181122.xml
```

Then convert:

```bash
python3 convert_to_csv.py                # all three -> original_dataset/{abp,semeval,checkthat}.csv
python3 convert_to_csv.py abp checkthat  # only the given ones
```

This writes `original_dataset/<corpus>.csv` with columns
`id,title,content,url,source,label,gold_label` (CheckThat has no url/source, so those
are blank). `label` is each corpus's raw label; `gold_label` is that label projected to
the common 3-class axis (`left`/`center`/`right`) — the field stage 1 reads. These
full-text CSVs are the input to stage 1 and are never committed.

---

## The released data (`dataset/*.csv`)

Each release file is **ID-only**: it carries no article text, so the copyrighted
bodies are never redistributed. Reattach the text locally by joining on `id`
(see *Reconstructing the full text*).

| column                 | description                                                            |
|------------------------|------------------------------------------------------------------------|
| `id`                   | item id; the join key to the original corpus                           |
| `gold_label`           | source/outlet label projected to 3 classes: `left` / `center` / `right`|
| `a_score`, `a_reason`  | judge A's Likert score (1–5) and its short rationale                   |
| `b_score`, `b_reason`  | judge B's Likert score and rationale                                   |
| `c_score`, `c_reason`  | judge C's Likert score and rationale                                   |
| `avg_score`            | mean of the valid judge scores (1–5)                                   |
| `tier`                 | `CLEAR` / `SUBTLE` / `AMBIGUOUS` / `MISMATCHED`                         |

Because every judge score, rationale, and the average are all released, and the tier
rule has no free parameters, you can **re-derive the tiers under your own thresholds**
by re-running `assign_tier.py` with different options.

### `dataset/*.judges.json`

Panel metadata for each corpus: the model/provider behind each judge letter
(`a`/`b`/`c`), the scale (`likert_1_5`), and the decoding settings
(`temperature`, `max_tokens`, `max_chars`, etc.). Example:

```json
{
  "judges": {
    "a": {"provider": "openrouter", "model": "google/gemini-3.1-flash-lite"},
    "b": {"provider": "openrouter", "model": "qwen/qwen3.6-flash"},
    "c": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"}
  },
  "scale": "likert_1_5", "temperature": 0.0, "max_tokens": 700, "max_chars": 0
}
```

### Tier definition

Let the panel scores be `s`, with mean `m` and range `r = max(s) - min(s)`:

| tier         | condition                     | meaning                                            |
|--------------|-------------------------------|----------------------------------------------------|
| `AMBIGUOUS`  | `r >= 2`                      | judges genuinely disagree                          |
| `CLEAR`      | `r < 2` and `m >= 4.5`        | label obvious on a surface read                    |
| `SUBTLE`     | `r < 2` and `3.5 <= m < 4.5`  | real leaning, visible only on a careful read       |
| `MISMATCHED` | `r < 2` and `m < 3.5`         | text does not support the label (mislabel candidate) |

The cutoffs `4.5`/`3.5` are the midpoints between adjacent Likert levels (5\|4, 4\|3),
i.e. properties of the measurement scale rather than tuned values; `r >= 2` is the
standard rater-disagreement notion. `CLEAR`+`SUBTLE` form the trusted pool.

---

## Reconstructing the full text

The release is keyed by `id`. To attach `title`/`content`, build the full-text CSVs
via *Getting the source corpora* (which produces `original_dataset/<corpus>.csv`) and
join on the same id:

```python
import csv, sys
csv.field_size_limit(sys.maxsize)

rel = {r["id"]: r for r in csv.DictReader(open("dataset/abp.csv", encoding="utf-8"))}
for row in csv.DictReader(open("original_dataset/abp.csv", encoding="utf-8")):
    if row["id"] in rel:
        rel[row["id"]]["title"]   = row["title"]
        rel[row["id"]]["content"] = row["content"]
```

The source corpora are ABP (Baly et al., 2020), CheckThat! 2023 Task 3A (Azizov et
al., 2023), and SemEval-2019 Task 4 (Kiesel et al., 2019); cite them alongside PoliCURE.

---

## Reproducing the pipeline

No third-party packages are required.

### Stage 1 — consistency scoring (`score_consistency.py`)

Scores a full-text CSV (columns `id`, `gold_label`, `title`, `content`) and writes
`*_scored.csv` plus a `*_scored.judges.json`. Configuration is in the `CONFIG` block
at the top of the file (there is no CLI).

> **Labels.** The scorer reads **`gold_label`** — the source label projected to the
> common 3-class axis, which `convert_to_csv.py` already writes (no separate
> harmonization step). It is the identity for ABP and CheckThat (already
> `left`/`center`/`right`); for SemEval's 5-way MBFC labels it maps `least → center`,
> `{left, left-center} → left`, `{right, right-center} → right`. The raw `label` column
> is kept alongside for provenance.

1. Provide API keys via **environment variables** (never hardcode them):

   ```bash
   export OPENROUTER_API_KEY=...     # or OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY
   ```

2. Edit the `CONFIG` block: `INPUT_FILES`, the three `RATER_*` models, and options
   (`TEMPERATURE=0.0`, `MAX_CHARS=0` for no truncation, `WORKERS`, `RESUME`, ...).
3. Run:

   ```bash
   python3 score_consistency.py
   ```

Scoring is resumable (`RESUME=True` merges prior scores by `id`) and checkpoints every
`SAVE_EVERY` rows. The judge prompt is fixed in `VERIFICATION_PROMPT`; decoding is
deterministic (`temperature=0`). For a key-free smoke test, set a rater's provider to
`mock`.

### Stage 2 — tier assignment (`assign_tier.py`)

```bash
python3 assign_tier.py --input abp_scored.csv
# optional per-tier accuracy report against subject predictions:
python3 assign_tier.py --input abp_scored.csv --validate abp_scored_infer.csv
```

Options: `--contest 2` (AMBIGUOUS range), `--clear 4.5`, `--subtle 3.5`,
`--output out.csv`. The rule uses only judge scores — subject predictions never enter
the tier definition, so validation is non-circular.

### Rebuilding the release files

The release `dataset/*.csv` are the `*_scored.csv` files with the text/provenance
columns dropped. To rebuild one (e.g. after regenerating `semeval_scored.csv`):

```python
import csv, sys
csv.field_size_limit(sys.maxsize)
KEEP = ["id", "gold_label", "a_score", "a_reason", "b_score", "b_reason",
        "c_score", "c_reason", "avg_score", "tier"]
with open("semeval_scored.csv", encoding="utf-8", newline="") as fi, \
     open("dataset/semeval.csv", "w", encoding="utf-8", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=KEEP, extrasaction="ignore"); w.writeheader()
    for row in csv.DictReader(fi):
        w.writerow({k: row.get(k, "") for k in KEEP})
```

#### SemEval: split parts and `merge_semeval.py`

`dataset/semeval.csv` (600,000 rows, ~449 MB) exceeds GitHub's 100 MB per-file limit,
so it is **not committed directly** (it is git-ignored). Instead it is committed as six
~75 MB parts under `dataset/semeval_parts/` (`semeval1.csv` ... `semeval6.csv`). After
cloning, rebuild the single file — standard library only, byte-identical to the
original — with:

```bash
python3 merge_semeval.py        # -> dataset/semeval.csv
```

(`abp.csv` and `checkthat.csv` are small enough to commit directly, so they need no
reassembly.) Alternatively, regenerate `semeval.csv` from the scored source
(`semeval_scored.csv`, ~3 GB, not committed) with the snippet above; with `pyarrow` the
column projection takes seconds:

```python
from pyarrow import csv as pacsv
KEEP = ["id", "gold_label", "a_score", "a_reason", "b_score", "b_reason",
        "c_score", "c_reason", "avg_score", "tier"]
t = pacsv.read_csv("semeval_scored.csv",
        convert_options=pacsv.ConvertOptions(include_columns=KEEP))
pacsv.write_csv(t.select(KEEP), "dataset/semeval.csv")   # optional: pip install pyarrow
```

---

## Requirements

- Python **3.8+**, standard library only (no `pip install`).
- Stage 1 needs network access and an API key for the chosen judge provider(s).
  Stage 2 and the release files need neither.

## License and responsible use

The **scripts** (`convert_to_csv.py`, `score_consistency.py`, `assign_tier.py`) are
released under the **MIT License** (`LICENSE`). The **curation artifacts** under
`dataset/` (tiers, judge scores/rationales, panel averages, `gold_label`, and the
judge metadata) are released under **CC BY 4.0** (`LICENSE-DATA`). Original article
text is **not** redistributed; it remains under the terms of the respective source
corpora, to which you must join by `id`.

The tiers and scores are produced by LLM judges and are intended for benchmarking and
research on media-bias measurement. They are **not** verdicts on any outlet or author,
and `gold_label` reflects the outlet-level rating inherited from the source corpus, not
ground truth about an individual article. Please use them accordingly and cite the
original corpora.

## Citation

If you use PoliCURE, please cite our paper and the three source corpora (ABP,
CheckThat!, SemEval-2019 Task 4).
