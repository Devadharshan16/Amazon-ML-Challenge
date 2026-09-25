"""
Step 2: Data Exploration
Run this once to understand the dataset before building the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"D:\New folder\Amazon-ML-Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset")

# ── Load all files ──────────────────────────────────────────────────────────
tr1 = pd.read_csv(BASE / "train/train_source1.tsv", sep="\t")
tr2 = pd.read_csv(BASE / "train/train_source2.tsv", sep="\t")
tr3 = pd.read_csv(BASE / "train/train_source3.tsv", sep="\t")
gt  = pd.read_csv(BASE / "train/train_ground_truth.tsv", sep="\t")

te1 = pd.read_csv(BASE / "test/test_source1.tsv", sep="\t")
te2 = pd.read_csv(BASE / "test/test_source2.tsv", sep="\t")
te3 = pd.read_csv(BASE / "test/test_source3.tsv", sep="\t")

print("=" * 60)
print("DATASET SIZES")
print("=" * 60)
print(f"Train Source1: {len(tr1):>6} records")
print(f"Train Source2: {len(tr2):>6} records")
print(f"Train Source3: {len(tr3):>6} records")
print(f"Ground Truth:  {len(gt):>6} rows")
print(f"Test  Source1: {len(te1):>6} records  ← must predict ALL of these")
print(f"Test  Source2: {len(te2):>6} records")
print(f"Test  Source3: {len(te3):>6} records")

print("\n" + "=" * 60)
print("COUNTRY DISTRIBUTION")
print("=" * 60)
for name, df in [("Train S1", tr1), ("Train S2", tr2), ("Train S3", tr3),
                 ("Test  S1", te1), ("Test  S2", te2), ("Test  S3", te3)]:
    counts = df["country"].value_counts().to_dict()
    print(f"{name}: {counts}")

print("\n" + "=" * 60)
print("GROUND TRUTH ANALYSIS")
print("=" * 60)
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
gt["num_matches"] = gt["matched_entity_ids"].apply(
    lambda x: len(x.split(",")) if x.strip() else 0
)
singletons = (gt["num_matches"] == 0).sum()
total = len(gt)
print(f"Total S1 entities in GT:    {total}")
print(f"Singletons (no match):      {singletons} ({100*singletons/total:.1f}%)")
print(f"Entities with ≥1 match:     {total - singletons} ({100*(total-singletons)/total:.1f}%)")
print(f"Match count distribution:")
print(gt["num_matches"].value_counts().sort_index().to_string())

# How many S2 vs S3 matches
all_matched = gt[gt["matched_entity_ids"] != ""]["matched_entity_ids"]
s2_count = sum(id.startswith("S2") for ids in all_matched for id in ids.split(","))
s3_count = sum(id.startswith("S3") for ids in all_matched for id in ids.split(","))
print(f"\nTotal S2 IDs matched: {s2_count}")
print(f"Total S3 IDs matched: {s3_count}")

print("\n" + "=" * 60)
print("SAMPLE NOISY RECORDS (first 5 S1 with matches)")
print("=" * 60)
gt_with_matches = gt[gt["num_matches"] > 0].head(5)
for _, row in gt_with_matches.iterrows():
    s1_id = row["source1_entity_id"]
    s1_rec = tr1[tr1["entity_id"] == s1_id].iloc[0]
    print(f"\nS1: [{s1_rec['entity_id']}] {s1_rec['business_name']} | {s1_rec['business_address']} | {s1_rec['country']}")
    for mid in row["matched_entity_ids"].split(",")[:3]:  # show first 3 matches
        src = tr2 if mid.startswith("S2") else tr3
        rec = src[src["entity_id"] == mid]
        if not rec.empty:
            r = rec.iloc[0]
            print(f"  ✓ [{r['entity_id']}] {r['business_name']} | {r['business_address']}")

print("\n" + "=" * 60)
print("NULL / MISSING VALUE CHECK")
print("=" * 60)
for name, df in [("Train S1", tr1), ("Train S2", tr2), ("Train S3", tr3)]:
    nulls = df.isnull().sum()
    print(f"{name}: {nulls.to_dict()}")

print("\n✅ Exploration complete!")
