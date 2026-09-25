# Agent Fix Instructions — `stage1_blocking.py`
## Amazon ML Challenge — Blocking Stage Bug Fixes

> **Read this fully before touching any code.**
> Three fixes are listed in priority order. Apply all three.
> After all fixes, re-run Stage 1 and check `blocking_recall.txt`. Target ≥ 97%.

---

## Fix 1 — CRITICAL: Replace `query_to_int_pairs` (Wrong Top-K Selection)

### What is wrong

The current implementation uses `np.unique(np.concatenate(lists))` to union all
candidate indices, then truncates with a plain slice `combined[:max_cands]`.

`np.unique` returns results sorted by **integer index** (i.e. by row number in the
source file), NOT by similarity. So for an S1 record named `"raj electronics"`,
blocking keeps S2 records with the lowest row numbers — e.g. rows 0 through 39 —
regardless of whether they share any tokens with `"raj electronics"`. The actual
best candidates (those sharing the most tokens) may be at rows 500, 2300, 8100,
etc., and are silently discarded.

This is the root cause of low blocking recall. It must be fixed before the recall
number means anything.

### What should happen instead

Count how many tokens each S23 candidate shares with the S1 query. Keep the
top-`max_cands` candidates by **shared token count**, not by row index.

### Exact replacement

**Delete** the entire existing `query_to_int_pairs` function (lines 85–110) and
**replace** it with the following:

```python
from collections import Counter

def query_to_int_pairs(s1_texts: np.ndarray, index: dict, token_fn, max_cands: int) -> tuple:
    """
    For each S1 text, find top-max_cands S23 candidates ranked by
    number of shared tokens (not by row index).
    Returns two int32 arrays: (s1_indices, s23_indices).
    """
    all_s1_int  = []
    all_s23_int = []

    for s1_idx, text in enumerate(tqdm(s1_texts, desc="    Querying",
                                       total=len(s1_texts), mininterval=2.0)):
        toks = [tok for tok in token_fn(text) if tok in index]
        if not toks:
            continue

        # Count shared tokens per S23 candidate
        counter = Counter()
        for tok in toks:
            for s23_idx in index[tok]:
                counter[s23_idx] += 1

        # Keep top-k by count (most similar first)
        top = [idx for idx, _ in counter.most_common(max_cands)]

        all_s1_int.extend([s1_idx] * len(top))
        all_s23_int.extend(top)

    if not all_s1_int:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    return np.array(all_s1_int, dtype=np.int32), np.array(all_s23_int, dtype=np.int32)
```

**Also add** the import at the top of the file (next to the existing imports):

```python
from collections import Counter
```

> Note: `Counter` is already imported via `collections` if `defaultdict` is imported
> from there. Add it explicitly: `from collections import defaultdict, Counter`.

---

## Fix 2 — Add Tamil/Hindi Script Diagnostic After Blocking Runs

### What is wrong

Records whose `business_name` is written in Tamil or Hindi script (e.g.
`"ரஜ இனவஸடமணடஸ எலஎலப"`) produce trigrams and word tokens that will **never
match** any English S1 name in blockers B1 or B2. The only blocker that can rescue
these pairs is B3 (address-based word blocking). However:

- `ADDR_MAX_LIST = 200` caps the address token index list length at 200.
- Common city names like `"chennai"`, `"mumbai"`, `"delhi"` appear in tens of
  thousands of records, so they are **filtered out** of the index entirely by this
  cap and contribute zero recall for script-switching pairs.

### What to do

**Step A — Run this diagnostic immediately after Stage 1 completes** (paste into a
throwaway script or a Jupyter cell):

```python
import pandas as pd
import pyarrow.parquet as pq
import re
from pathlib import Path

CACHE = Path(r"D:\New folder\Amazon-ML-Challenge\submission\code\business_entity_resolution\src\cache")

cands = pq.read_table(CACHE / 'train_candidates.parquet').to_pandas()
s2    = pd.read_parquet(CACHE / 'train_s2_clean.parquet')
s3    = pd.read_parquet(CACHE / 'train_s3_clean.parquet')

# Tamil Unicode block: U+0B80–U+0BFF
# Hindi (Devanagari) block: U+0900–U+097F
SCRIPT_RE = re.compile(r'[\u0B80-\u0BFF\u0900-\u097F]')

tamil_hindi_s2 = s2[s2['business_name'].str.contains(SCRIPT_RE, na=False)]
tamil_hindi_s3 = s3[s3['business_name'].str.contains(SCRIPT_RE, na=False)]

in_cands_s2 = tamil_hindi_s2['entity_id'].isin(cands['candidate_id'].dropna())
in_cands_s3 = tamil_hindi_s3['entity_id'].isin(cands['candidate_id'].dropna())

print(f"Tamil/Hindi S2 records: {len(tamil_hindi_s2):,}")
print(f"  → In candidates:      {in_cands_s2.sum():,}  ({100*in_cands_s2.mean():.1f}%)")
print(f"Tamil/Hindi S3 records: {len(tamil_hindi_s3):,}")
print(f"  → In candidates:      {in_cands_s3.sum():,}  ({100*in_cands_s3.mean():.1f}%)")
```

**Step B — Interpret the result:**

| Result | Action |
|---|---|
| ≥ 80% of script records appear in candidates | B3 is working. No change needed. |
| < 80% of script records appear in candidates | Apply the config change below. |

**Step C — If the diagnostic shows < 80%, change these two constants** in the config
section at the top of `stage1_blocking.py`:

```python
# OLD
ADDR_MAX_LIST  = 200
MAX_ADDR_CANDS = 20

# NEW
ADDR_MAX_LIST  = 2000   # allow common city names into the address index
MAX_ADDR_CANDS = 40     # retrieve more address-based candidates per S1
```

Then re-run Stage 1 (training split only first, check recall, then test).

---

## Fix 3 — Raise `WORD_MAX_LIST` If Recall Is Below 97%

### What is wrong

`WORD_MAX_LIST = 300` means any word token that appears in more than 300 S23 records
is dropped from the word index entirely. For short business names with only one or
two significant tokens (e.g. `"raj"` after suffix stripping), the true match may
be record number 301+ in that word's posting list and is permanently unreachable.

### When to apply this fix

**Only if `blocking_recall.txt` reports < 97% after Fixes 1 and 2.**
If recall is already ≥ 97%, skip this — raising the cap costs memory and runtime
for no gain.

### Change to make

```python
# OLD
WORD_MAX_LIST = 300

# NEW
WORD_MAX_LIST = 1000
```

After changing, re-run blocking for the training split and re-check recall.

> If recall is still < 97% after this change, the remaining misses are likely
> cross-script pairs (Tamil/Hindi names with no English address). Those require a
> separate transliteration approach and are lower priority given time constraints.
> Move to Stage 2 rather than spending more time on blocking.

---

## Re-run Order After Applying All Fixes

Run in this exact sequence:

```bash
# 1. Training split blocking + recall check
python stage1_blocking.py
# → reads cache/train_s*_clean.parquet
# → writes cache/train_candidates.parquet
# → writes cache/blocking_recall.txt   ← check this number

# 2. Tamil/Hindi diagnostic (Step A from Fix 2 above)
python tamil_diagnostic.py   # or paste into Jupyter

# 3. If recall < 97% → apply Fix 3 config change → re-run step 1

# 4. Once recall ≥ 97%, run test split blocking
# (blocking_recall.txt will still show only train recall — that's fine)
```

`stage1_blocking.py` already runs both train and test splits sequentially in
`__main__`. If you re-run the full script, it overwrites both output parquets.
That is fine — just make sure training recall is confirmed ≥ 97% before you move
on.

---

## What NOT to Change

Do not modify any of the following — they are correct as written:

- `stage0_clean.py` — cleaning logic is correct and country-agnostic
- `build_trigram_index_s1_only` — the S1-bounded trigram index is the right
  optimization; do not revert it to a full index
- `write_pairs_chunked` and `write_singletons` — streaming-to-disk approach is
  correct; do not accumulate pairs in memory
- `block_country` singleton tracking — `s1_has_cand` correctly persists across
  both S2 and S3 calls; do not split it
- Country-partition logic in `run_blocking` — correct, France-safe

---

## Success Criteria Before Moving to Stage 2

| Check | Target |
|---|---|
| `blocking_recall.txt` | ≥ 97.00% |
| Tamil/Hindi script records in candidates | ≥ 80% |
| `train_candidates.parquet` exists and is non-empty | ✅ |
| `test_candidates.parquet` exists and is non-empty | ✅ |

Once all four are green, **stop touching blocking** and move immediately to Stage 2
(feature engineering). Every hour spent over-engineering blocking past 97% is an
hour not spent on the LightGBM model and threshold tuning, which is where the F₀.₅
score actually comes from.
