"""
Stage 0: Data Cleaning & Normalization
Loads all TSV files, normalizes text, and saves cleaned versions.

Key operations:
- Lowercase, strip punctuation
- Expand common abbreviations (name + address)
- Strip legal suffixes → separate 'name_stripped' field
- Normalize unicode (handles Tamil/Hindi script → keep as-is for char n-gram TF-IDF,
  but also build a romanized-safe version)
- Handle NaN addresses (fill with "")
- Never gate logic on specific country values
"""

import re
import unicodedata
import pandas as pd
from pathlib import Path
import joblib
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DATA = Path(r"D:\New folder\Amazon-ML-Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset")
OUT_DIR   = Path(r"D:\New folder\Amazon-ML-Challenge\submission\code\business_entity_resolution\src\cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Abbreviation Maps ──────────────────────────────────────────────────────────
# Applied to both name and address (opportunistically — no country gating)
NAME_ABBREVS = {
    r'\bcorp\b':          'corporation',
    r'\bpvt\b':           'private',
    r'\bltd\b':           'limited',
    r'\bllp\b':           'limited liability partnership',
    r'\bllc\b':           'limited liability company',
    r'\binc\b':           'incorporated',
    r'\bco\b':            'company',
    r'\bbros\b':          'brothers',
    r'\basso?c\b':        'associates',
    r'\bentps?\b':        'enterprises',
    r'\bmnfg?\b':         'manufacturing',
    r'\bindus?\b':        'industries',
    r'\bservs?\b':        'services',
    r'\bintl\b':          'international',
    r'\bnational\b':      'national',
    r'&':                 'and',
}

ADDR_ABBREVS = {
    r'\brd\b':            'road',
    r'\bst\b':            'street',
    r'\bave?\b':          'avenue',
    r'\bblvd\b':          'boulevard',
    r'\bdr\b':            'drive',
    r'\bln\b':            'lane',
    r'\bct\b':            'court',
    r'\bpl\b':            'place',
    r'\bsq\b':            'square',
    r'\bhwy\b':           'highway',
    r'\bpky?\b':          'parkway',
    r'\bapt\b':           'apartment',
    r'\bste\b':           'suite',
    r'\bflr?\b':          'floor',
    r'\bnr\b':            'near',
    r'\bopp\b':           'opposite',
    r'\bn\b':             'north',
    r'\bs\b':             'south',
    r'\be\b':             'east',
    r'\bw\b':             'west',
    r'&':                 'and',
}

# Legal suffixes to strip for the 'name_stripped' field
LEGAL_SUFFIXES = [
    r'\s+pvt\.?\s+ltd\.?$',
    r'\s+private\s+limited$',
    r'\s+ltd\.?$',
    r'\s+limited$',
    r'\s+llp$',
    r'\s+llc$',
    r'\s+inc\.?$',
    r'\s+incorporated$',
    r'\s+corp\.?$',
    r'\s+corporation$',
    r'\s+co\.?$',
    r'\s+company$',
    r'\s+& co\.?$',
]
LEGAL_SUFFIX_RE = re.compile('|'.join(LEGAL_SUFFIXES), re.IGNORECASE)


def apply_abbrevs(text: str, abbrev_map: dict) -> str:
    """Apply regex abbreviation expansions."""
    for pattern, replacement in abbrev_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_text(text: str) -> str:
    """
    Core normalization:
    - Fill NaN
    - Lowercase
    - Strip extra whitespace and punctuation (keep alphanumeric + space)
    - Collapse multiple spaces
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower().strip()
    # Remove punctuation except spaces (keep unicode letters/digits for Tamil/Hindi)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_name(name: str) -> str:
    """Normalize business name."""
    text = normalize_text(name)
    text = apply_abbrevs(text, NAME_ABBREVS)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def strip_legal_suffix(name: str) -> str:
    """Remove legal suffix from a cleaned name."""
    return LEGAL_SUFFIX_RE.sub('', name).strip()


def clean_address(addr: str) -> str:
    """Normalize address string."""
    text = normalize_text(addr)
    text = apply_abbrevs(text, ADDR_ABBREVS)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to a source dataframe."""
    df = df.copy()

    print("  → Cleaning names...")
    df['name_clean']    = df['business_name'].apply(clean_name)
    df['name_stripped'] = df['name_clean'].apply(strip_legal_suffix)

    print("  → Cleaning addresses...")
    df['address_clean'] = df['business_address'].apply(clean_address)

    # Country as-is (lowercase, treat as free-form string — no encoding)
    df['country_clean'] = df['country'].str.lower().str.strip()

    # Combined field for TF-IDF blocking (name + address together)
    df['name_addr_combined'] = df['name_clean'] + ' ' + df['address_clean']

    return df


def load_and_clean_all():
    files = {
        'train_s1': BASE_DATA / 'train/train_source1.tsv',
        'train_s2': BASE_DATA / 'train/train_source2.tsv',
        'train_s3': BASE_DATA / 'train/train_source3.tsv',
        'test_s1':  BASE_DATA / 'test/test_source1.tsv',
        'test_s2':  BASE_DATA / 'test/test_source2.tsv',
        'test_s3':  BASE_DATA / 'test/test_source3.tsv',
    }

    cleaned = {}
    for key, path in files.items():
        print(f"\nProcessing {key} ({path.name})...")
        df = pd.read_csv(path, sep='\t', dtype=str)
        df = process_df(df)
        out_path = OUT_DIR / f"{key}_clean.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  ✅ Saved → {out_path.name}  ({len(df):,} rows)")
        cleaned[key] = df

    # Also load ground truth (no cleaning needed, just save)
    gt = pd.read_csv(BASE_DATA / 'train/train_ground_truth.tsv', sep='\t', dtype=str)
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    gt.to_parquet(OUT_DIR / 'ground_truth.parquet', index=False)
    print(f"\n✅ Ground truth saved. ({len(gt):,} rows)")

    return cleaned


if __name__ == '__main__':
    print("=" * 60)
    print("Stage 0: Data Cleaning")
    print("=" * 60)
    cleaned = load_and_clean_all()

    # Quick sanity check
    s1 = cleaned['train_s1']
    print("\n=== SAMPLE CLEANED RECORDS (Train S1) ===")
    for _, row in s1.head(3).iterrows():
        print(f"  raw name:     {row['business_name']}")
        print(f"  clean name:   {row['name_clean']}")
        print(f"  stripped:     {row['name_stripped']}")
        print(f"  raw addr:     {row['business_address']}")
        print(f"  clean addr:   {row['address_clean']}")
        print()

    print("\n✅ Stage 0 complete! Cleaned files saved to cache/")
