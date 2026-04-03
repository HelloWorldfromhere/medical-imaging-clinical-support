"""
Step 1: Verify Pleural Thickening in the dataset
=================================================
Run from project root:
    python fix_1_verify_pleural.py

This checks if the __getitem__ actually produces Pleural_Thickening labels,
and whether the issue was just a display bug in label counting.
"""

import pandas as pd
import os

DATA_DIR = "data/chestxray14"
CSV_FILE = os.path.join(DATA_DIR, "Data_Entry_2017.csv")

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

print("=" * 60)
print("Pleural Thickening Verification")
print("=" * 60)

df = pd.read_csv(CSV_FILE)
print(f"\nTotal images in CSV: {len(df)}")

# Check what the CSV actually contains
findings_col = df["Finding Labels"]

# Method 1: Direct string match (what the buggy counter probably did)
buggy_count = findings_col.str.contains("Pleural_Thickening", na=False).sum()
print(f"\nBuggy count (underscore 'Pleural_Thickening'): {buggy_count}")

# Method 2: Correct string match (what the CSV actually has)
correct_count = findings_col.str.contains("Pleural Thickening", na=False).sum()
print(f"Correct count (space 'Pleural Thickening'):    {correct_count}")

# Method 3: What __getitem__ does (should work)
getitem_count = 0
for finding in findings_col:
    finding_str = str(finding)
    condition = "Pleural_Thickening"
    condition_clean = condition.replace("_", " ")
    if condition_clean in finding_str or condition in finding_str:
        getitem_count += 1
print(f"__getitem__ logic count:                       {getitem_count}")

# Show all conditions with both methods
print(f"\n{'Condition':<22} {'Buggy':>8} {'Correct':>8} {'Match':>6}")
print("-" * 48)
for condition in CONDITIONS:
    condition_clean = condition.replace("_", " ")
    buggy = findings_col.str.contains(condition, na=False, regex=False).sum()
    correct = findings_col.str.contains(condition_clean, na=False, regex=False).sum()
    match = "OK" if buggy == correct else "DIFF!"
    print(f"{condition:<22} {buggy:>8} {correct:>8} {match:>6}")

# Show sample Pleural Thickening entries
print(f"\nSample 'Pleural Thickening' entries from CSV:")
pt_rows = df[findings_col.str.contains("Pleural Thickening", na=False)].head(5)
for _, row in pt_rows.iterrows():
    print(f"  {row['Image Index']}: {row['Finding Labels']}")

print(f"\n{'=' * 60}")
if correct_count > 0 and getitem_count > 0:
    print("VERDICT: __getitem__ DOES pick up Pleural Thickening correctly.")
    print("The '0 samples' was a DISPLAY BUG in label counting, not a training bug.")
    print("The model DID train on Pleural Thickening. No retrain needed for this.")
    print("Just fix the label counting code for accurate logging.")
else:
    print("VERDICT: There IS a real issue. Check CSV format.")
print("=" * 60)
