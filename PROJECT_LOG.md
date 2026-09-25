# Project Log — Amazon ML Challenge: Business Entity Resolution

> Maintained for the documentation team.
> Each entry = one step we completed. Written in simple language.

---

## Step 1 — Set Up Folder Structure

**What we did:**
Created the required submission folder structure as specified in the problem statement.

**Folders created:**
```
Amazon-ML-Challenge/
└── submission/
    ├── output/                                          ← final result files go here
    └── code/
        └── business_entity_resolution/
            ├── src/                                     ← all our Python scripts go here
            ├── README.md                                ← how to run the pipeline
            └── requirements.txt                        ← list of Python packages needed
```

**Why:**
The problem statement requires this exact structure in the final zip submission. We built it first so every file we create goes in the right place.

---

## Step 1.5 — Install Python Packages

**File:** `submission/code/business_entity_resolution/requirements.txt`

**What it contains:**
A list of all external Python libraries our pipeline needs.

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Load TSV files, handle data |
| `rapidfuzz` | Fast string similarity (Levenshtein, Jaro-Winkler, token matching) |
| `metaphone` | Phonetic matching — catches "Sharma" vs "Sarma" type variants |
| `lightgbm` | Our main ML model (gradient boosting — fast and accurate) |
| `xgboost` | Backup ML model (same idea as LightGBM) |
| `scikit-learn` | TF-IDF vectorizer for blocking |
| `sparse_dot_topn` | Fast approximate nearest neighbor search on TF-IDF vectors |
| `tqdm` | Progress bars |
| `joblib` | Save/load trained model to disk |
| `pyarrow` | Read/write parquet files (faster than CSV/TSV for large data) |

**Why:**
Instead of installing packages one by one, anyone can run `pip install -r requirements.txt` to set up the full environment in one command.

---

## Step 2 — Data Exploration

**File:** `submission/code/business_entity_resolution/src/explore_data.py`

**What it does:**
Reads all 6 dataset files and prints statistics to understand the data before building anything.

**Simple example of what it showed:**
```
Train Source1: 2,206,821 records
Train Source2: 5,034,616 records
Train Source3: 5,285,603 records

Countries in Test: US, India, France  ← France NOT in training data!

Singletons (no match): 5.6%
Entities with matches: 94.4%

Sample noisy pair:
  S1: "Raj Investments LLP"
  S2: "ரஜ இனவஸடமணடஸ எலஎலப"   ← same business, written in Tamil script!
```

**Key findings:**
1. Dataset is huge — 22+ trillion possible pairs. Cannot compare everything. Need smart filtering (blocking).
2. France appears only in test (not in training). Must not hardcode country logic.
3. 94.4% of S1 records have at least one match (mostly 2–4 matches each).
4. Noise types found: Tamil/Hindi script names, typos, address reordering, missing addresses, legal suffix variations.

**Why:**
Understanding the data before writing any pipeline code. Every design decision in the next steps comes from what we found here.

---

## Step 3 — Stage 0: Data Cleaning

**File:** `submission/code/business_entity_resolution/src/stage0_clean.py`

**What it does:**
Reads all 6 raw TSV files, cleans and normalizes every record, and saves cleaned versions as `.parquet` files in a `cache/` folder.

**Cleaning steps applied to each record:**

| Step | Before | After |
|---|---|---|
| Lowercase | `Raj Investments LLP` | `raj investments llp` |
| Expand abbreviations | `raj investments llp` | `raj investments limited liability partnership` |
| Strip punctuation | `C.I.T. Colony, 2Nd` | `c i t colony 2nd` |
| Strip legal suffix | `raj investments limited liability partnership` | `raj investments` ← saved as `name_stripped` |
| Handle missing address | `NaN` | `""` (empty string) |
| Combined field | — | `"raj investments limited... c i t colony..."` ← name + address together |

**Output files (saved to `src/cache/`):**
```
train_s1_clean.parquet    (2.2M rows)
train_s2_clean.parquet    (5.0M rows)
train_s3_clean.parquet    (5.3M rows)
test_s1_clean.parquet     (1.7M rows)
test_s2_clean.parquet     (4.9M rows)
test_s3_clean.parquet     (5.1M rows)
ground_truth.parquet      (2.2M rows)
```

**Why:**
- Raw data has abbreviations, punctuation, and inconsistent casing that make "Pvt Ltd" and "Private Limited" look completely different to a computer. Cleaning makes them identical.
- `name_stripped` removes legal suffixes so "Raj Investments LLP" and "Raj Investments" can be recognized as the same business during blocking.
- Parquet format loads ~10x faster than TSV — important since we re-load these files multiple times in later stages.
- All cleaning is country-agnostic (no `if country == "India"` logic) so it works for France too.

---

---

## Step 4 — Stage 1: Blocking / Candidate Generation

**File:** `submission/code/business_entity_resolution/src/stage1_blocking.py`

**What it does:**
For each S1 record, finds a small shortlist of S2/S3 records that are *likely* to be a match — without comparing every possible pair (which is impossible at our scale).

**Why blocking is needed:**
```
Without blocking: 2.2M S1 × 10.3M (S2+S3) = 22 TRILLION pairs to check → impossible
With blocking:    ~2.2M S1 × ~30 candidates = ~66 MILLION pairs to check → manageable
```

**Three blocking strategies used (results are combined/unioned):**

> **Note on implementation:** We originally planned to use TF-IDF matrix multiplication (ANN), but at 4M+ records it was too slow (would take hours per country). We rewrote it using an **inverted index** approach — same idea, 10–20x faster.

1. **Char trigram inverted index on `name_stripped`** *(Primary — catches typos & abbreviations)*

   Build a lookup table: `{3-char substring → [list of S2/S3 entity_ids]}`
   For each S1, count how many trigrams it shares with each S2/S3. Keep top-30 by count.

   *Example:*
   ```
   S1 name: "dahlia power reliable"
   Trigrams: ["dah", "ahl", "hli", "lia", "pow", "owe", "wer", ...]
   S2 name:  "dahlia ponr reliable"
   Shared:   "dah", "ahl", "hli", "lia", "rel", "eli"... → high overlap → candidate ✅
   ```
   Trigrams appearing in >2000 records are automatically skipped — they are too common to be useful.

2. **Word token inverted index on `name_stripped`** *(Safety net — exact/near-exact word matches)*

   Build: `{word → [list of S2/S3 entity_ids]}`
   For each S1, collect ALL S2/S3 sharing at least one significant word.

   *Example:* S1 has `"raj"` → finds all S2/S3 records also containing the word `"raj"`.
   Words appearing in >5000 records (e.g., "traders", "foods") are skipped.

3. **Word token inverted index on `address_clean`** *(Cross-script rescue — catches Hindi/Tamil-script names via their English address)*

   Same as B2 but on the address field. Critical because some Indian records have names written in
   Tamil or Hindi script (unreadable by the other blockers) but English addresses.

   *Example:*
   - S1: `"raj investments"` | address: `"c i t colony mylapore chennai"`
   - S2: `"ரஜ இனவஸடமணடஸ"` (Tamil script — name doesn't match anything) | same English address
   - B3 finds this pair because the address words match ✅

**All blocking is done within country partitions** — US records are never compared to India or France records. This alone eliminates 95%+ of impossible comparisons.

**Output saved to `src/cache/`:**
```
train_candidates.parquet   ← (s1_id, candidate_id) one row per candidate pair, train set
test_candidates.parquet    ← (s1_id, candidate_id) one row per candidate pair, test set
blocking_recall.txt        ← % of true matches found in candidates (target: ≥97%)
```

**Why we care about the recall number:**
If our blocking misses a true match, no ML model can recover it later — it's a hard ceiling on our score. We need ≥97% recall. If below that, we loosen the blocking parameters (lower the cap threshold or raise top-K) and re-run.

---

<!-- Future steps will be added below as we complete them -->
