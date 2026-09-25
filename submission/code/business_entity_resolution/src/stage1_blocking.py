"""
Stage 1: Blocking / Candidate Generation  (v5 — Streaming to Disk)

Root cause of previous crash (silent exit):
  - Pandas DataFrame creation with 50-60 million string pairs (India: 50M, US: ~67M)
    takes ~10-15GB of RAM, exceeding system limits.
  
Fix in v5:
  - Keep everything as integers internally as before.
  - Stream the final candidate pairs directly to disk (ParquetWriter) in chunks.
  - Never hold the full dataset in memory as a Pandas DataFrame.
  - No `pd.concat` of millions of string rows.

Expected peak memory: ~1.5 GB
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

CACHE = Path(r"D:\New folder\Amazon-ML-Challenge\submission\code\business_entity_resolution\src\cache")

# ── Config ─────────────────────────────────────────────────────────────────────
CHAR_NGRAM_N       = 3    
WORD_MAX_LIST      = 300  
ADDR_MAX_LIST      = 200  
CHAR_MAX_LIST      = 200  
MAX_WORD_CANDS     = 40   
MAX_CHAR_CANDS     = 20   
MAX_ADDR_CANDS     = 20   
MIN_WORD_LEN       = 3

GENERIC_NAME = {
    'and', 'the', 'of', 'in', 'for', 'co', 'ltd', 'llc', 'llp', 'inc',
    'pvt', 'corp', 'company', 'limited', 'incorporated', 'corporation',
    'private', 'enterprises', 'traders', 'foods', 'services', 'solutions',
    'group', 'industries', 'international', 'associates', 'brothers',
    'san', 'null', 'nan',
}
GENERIC_ADDR = {'and', 'the', 'of', 'in', 'null', 'nan', 'near', 'opp'}


# ══════════════════════════════════════════════════════════════════════════════
def word_tokens(text: str, generic: set) -> list:
    if not isinstance(text, str) or not text.strip(): return []
    return [t for t in text.split() if len(t) >= MIN_WORD_LEN and t not in generic]

def char_ngrams(text: str, n: int = CHAR_NGRAM_N) -> list:
    if not isinstance(text, str) or len(text) < n: return []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def build_index(texts: np.ndarray, token_fn, max_list: int) -> dict:
    raw = defaultdict(list)
    for idx, text in enumerate(tqdm(texts, desc="    Building index", mininterval=2.0)):
        for tok in set(token_fn(text)):
            raw[tok].append(idx)
    index = {k: np.array(v, dtype=np.int32) for k, v in raw.items() if 0 < len(v) <= max_list}
    del raw; gc.collect()
    return index

def build_trigram_index_s1_only(s1_texts: np.ndarray, s23_texts: np.ndarray, max_list: int) -> dict:
    print("    Collecting S1 trigrams...")
    s1_tg_set = set()
    for text in s1_texts: s1_tg_set.update(char_ngrams(text))
    
    print("    Building S23 trigram index (S1-bounded)...")
    raw = defaultdict(list)
    for idx, text in enumerate(tqdm(s23_texts, desc="    Indexing S23", mininterval=2.0)):
        for tg in set(char_ngrams(text)):
            if tg in s1_tg_set:
                raw[tg].append(idx)

    index = {k: np.array(v, dtype=np.int32) for k, v in raw.items() if 0 < len(v) <= max_list}
    del raw, s1_tg_set; gc.collect()
    return index

def query_to_int_pairs(s1_texts: np.ndarray, index: dict, token_fn, max_cands: int) -> tuple:
    n_s1 = len(s1_texts)
    cap  = n_s1 * max_cands
    s1_buf  = np.empty(cap, dtype=np.int32)
    s23_buf = np.empty(cap, dtype=np.int32)
    ptr = 0

    for s1_idx, text in enumerate(tqdm(s1_texts, desc="    Querying", total=n_s1, mininterval=2.0)):
        lists = [index[tok] for tok in token_fn(text) if tok in index]
        if not lists: continue

        combined = np.unique(np.concatenate(lists))
        if len(combined) > max_cands: combined = combined[:max_cands]

        n = len(combined)
        if ptr + n > cap:
            extra = n_s1 * max_cands
            s1_buf  = np.concatenate([s1_buf,  np.empty(extra, dtype=np.int32)])
            s23_buf = np.concatenate([s23_buf, np.empty(extra, dtype=np.int32)])
            cap += extra

        s1_buf[ptr:ptr+n]  = s1_idx
        s23_buf[ptr:ptr+n] = combined
        ptr += n

    return s1_buf[:ptr], s23_buf[:ptr]

# ══════════════════════════════════════════════════════════════════════════════
def write_pairs_chunked(s1_ids: np.ndarray, src_ids: np.ndarray, pairs_int: np.ndarray, writer: pq.ParquetWriter):
    """Convert int arrays to strings in chunks of 5M and write to parquet to save memory."""
    chunk_size = 5_000_000
    n_pairs = len(pairs_int)
    
    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        chunk = pairs_int[start:end]
        
        s1_chunk = s1_ids[chunk[:, 0]]
        src_chunk = src_ids[chunk[:, 1]]
        
        table = pa.Table.from_arrays(
            [pa.array(s1_chunk), pa.array(src_chunk)],
            names=['s1_id', 'candidate_id']
        )
        writer.write_table(table)
        
        del s1_chunk, src_chunk, table, chunk; gc.collect()

def write_singletons(s1_ids_with_no_cands, writer: pq.ParquetWriter):
    if len(s1_ids_with_no_cands) == 0: return
    table = pa.Table.from_arrays(
        [
            pa.array(s1_ids_with_no_cands, type=pa.large_string()), 
            pa.array([None] * len(s1_ids_with_no_cands), type=pa.large_string())
        ],
        names=['s1_id', 'candidate_id']
    )
    writer.write_table(table)


def block_one_source(s1_df: pd.DataFrame, src_df: pd.DataFrame, src_name: str, writer: pq.ParquetWriter, s1_has_cand: set):
    print(f"\n    ── vs {src_name} ({len(src_df):,} records) ──")

    src_ids   = src_df['entity_id'].values
    src_names = src_df['name_stripped'].fillna('').values
    src_addrs = src_df['address_clean'].fillna('').values

    s1_ids    = s1_df['entity_id'].values
    s1_names  = s1_df['name_stripped'].fillna('').values
    s1_addrs  = s1_df['address_clean'].fillna('').values

    all_s1_int  = []
    all_src_int = []

    # ── B1
    print("    [B1] Word token blocking on name...")
    idx = build_index(src_names, lambda t: word_tokens(t, GENERIC_NAME), WORD_MAX_LIST)
    s1i, s23i = query_to_int_pairs(s1_names, idx, lambda t: word_tokens(t, GENERIC_NAME), MAX_WORD_CANDS)
    all_s1_int.append(s1i); all_src_int.append(s23i)
    del idx, s1i, s23i; gc.collect()

    # ── B2
    print("    [B2] Char trigram blocking on name...")
    idx = build_trigram_index_s1_only(s1_names, src_names, CHAR_MAX_LIST)
    s1i, s23i = query_to_int_pairs(s1_names, idx, char_ngrams, MAX_CHAR_CANDS)
    all_s1_int.append(s1i); all_src_int.append(s23i)
    del idx, s1i, s23i; gc.collect()

    # ── B3
    print("    [B3] Word token blocking on address...")
    addr_mask    = np.array([len(a) > 3 for a in src_addrs])
    s1_addr_mask = np.array([len(a) > 3 for a in s1_addrs])

    if addr_mask.any() and s1_addr_mask.any():
        src_addr_idx = np.where(addr_mask)[0].astype(np.int32)
        idx_local = build_index(src_addrs[addr_mask], lambda t: word_tokens(t, GENERIC_ADDR), ADDR_MAX_LIST)
        idx_global = {k: src_addr_idx[v] for k, v in idx_local.items()}
        del idx_local, src_addr_idx; gc.collect()

        s1_addr_idx = np.where(s1_addr_mask)[0].astype(np.int32)
        s1i_local, s23i = query_to_int_pairs(s1_addrs[s1_addr_mask], idx_global, lambda t: word_tokens(t, GENERIC_ADDR), MAX_ADDR_CANDS)
        
        s1i = s1_addr_idx[s1i_local]
        all_s1_int.append(s1i); all_src_int.append(s23i)
        del idx_global, s1i_local, s1i, s23i, s1_addr_idx; gc.collect()

    # ── Union
    print("    Combining blockers...")
    if not all_s1_int:
        return 0

    s1_combined  = np.concatenate(all_s1_int)
    src_combined = np.concatenate(all_src_int)
    del all_s1_int, all_src_int; gc.collect()

    pairs = np.stack([s1_combined, src_combined], axis=1)
    del s1_combined, src_combined; gc.collect()
    pairs = np.unique(pairs, axis=0)
    
    # Mark S1 entities that have candidates
    for i in np.unique(pairs[:, 0]):
        s1_has_cand.add(s1_ids[i])

    print(f"    Writing {len(pairs):,} candidate pairs to disk...")
    write_pairs_chunked(s1_ids, src_ids, pairs, writer)
    
    num_pairs = len(pairs)
    del pairs, s1_ids, src_ids, s1_names, src_names, s1_addrs, src_addrs; gc.collect()
    return num_pairs

def block_country(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame, country: str, writer: pq.ParquetWriter):
    print(f"\n  ── Country: {country.upper()} ────────────────────────────────────────")
    s1_has_cand = set()
    total_pairs = 0

    if len(s2_df) > 0:
        total_pairs += block_one_source(s1_df, s2_df, "S2", writer, s1_has_cand)
    if len(s3_df) > 0:
        total_pairs += block_one_source(s1_df, s3_df, "S3", writer, s1_has_cand)

    # Write singletons
    no_cand = [eid for eid in s1_df['entity_id'].values if eid not in s1_has_cand]
    write_singletons(no_cand, writer)
    
    del s1_has_cand; gc.collect()
    print(f"\n  ✅ {country.upper()} done: {total_pairs:,} pairs + {len(no_cand):,} singletons")

# ══════════════════════════════════════════════════════════════════════════════
def run_blocking(split: str, out_path: Path):
    print(f"\n{'='*60}")
    print(f"BLOCKING — {split.upper()}")
    print(f"{'='*60}")

    s1 = pd.read_parquet(CACHE / f"{split}_s1_clean.parquet")
    s2 = pd.read_parquet(CACHE / f"{split}_s2_clean.parquet")
    s3 = pd.read_parquet(CACHE / f"{split}_s3_clean.parquet")
    
    schema = pa.schema([('s1_id', pa.large_string()), ('candidate_id', pa.large_string())])
    
    with pq.ParquetWriter(out_path, schema) as writer:
        for country in sorted(s1['country_clean'].dropna().unique()):
            s1_c = s1[s1['country_clean'] == country].reset_index(drop=True)
            s2_c = s2[s2['country_clean'] == country].reset_index(drop=True)
            s3_c = s3[s3['country_clean'] == country].reset_index(drop=True)
            block_country(s1_c, s2_c, s3_c, country, writer)
            gc.collect()

    del s1, s2, s3; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
def measure_recall(out_path: Path):
    print(f"\n{'='*60}")
    print("Measuring blocking recall...")
    print(f"{'='*60}")

    gt = pd.read_parquet(CACHE / 'ground_truth.parquet')
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    
    print("Loading candidate pairs for recall check...")
    # Load just the needed columns
    cands = pq.read_table(out_path).to_pandas()
    cands = cands.dropna(subset=['candidate_id'])
    
    cand_lookup = cands.groupby('s1_id')['candidate_id'].apply(set).to_dict()
    del cands; gc.collect()

    total, found, missed = 0, 0, []
    for _, row in tqdm(gt.iterrows(), total=len(gt), desc="Checking recall"):
        s1_id   = row['source1_entity_id']
        matched = row['matched_entity_ids']
        if not matched or not isinstance(matched, str) or not matched.strip(): continue
        
        cands_set = cand_lookup.get(s1_id, set())
        for tid in [m.strip() for m in matched.split(',') if m.strip()]:
            total += 1
            if tid in cands_set:
                found += 1
            elif len(missed) < 10:
                missed.append((s1_id, tid))

    recall = found / total if total > 0 else 0
    print(f"\n  True pairs:          {total:,}")
    print(f"  Found in candidates: {found:,}")
    print(f"  Blocking recall:     {recall*100:.2f}%  (target ≥ 97%)")
    
    with open(CACHE / 'blocking_recall.txt', 'w') as f:
        f.write(f"Blocking Recall: {recall*100:.2f}%\n")
        f.write(f"True pairs:  {total:,}\nFound:       {found:,}\n")
    return recall

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    train_out = CACHE / 'train_candidates.parquet'
    run_blocking('train', train_out)
    recall = measure_recall(train_out)
    gc.collect()

    test_out = CACHE / 'test_candidates.parquet'
    run_blocking('test', test_out)
    print(f"\n✅ Stage 1 complete! Blocking recall: {recall*100:.2f}%")
