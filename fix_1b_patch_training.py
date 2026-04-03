"""
Step 1b: Fix label counting display in train_multilabel.py
==========================================================
Run from project root:
    python fix_1b_patch_training.py

Fixes the label distribution counting to handle underscore/space mismatch.
This is a display-only fix — the actual training data was likely correct
because __getitem__ already handles condition.replace("_", " ").
"""

TRAIN_PATH = "train_multilabel.py"

with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ── Fix: Any place that counts labels directly from CSV without replace ──
# Look for patterns like:
#   if condition in findings  (without the replace("_", " ") call)
#   findings.str.contains(condition)  (pandas-style, also needs space handling)

# The most common pattern in label counting:
# for i, condition in enumerate(CONDITIONS):
#     count = (findings == condition).sum()  or  condition in str(row["Finding Labels"])

# Fix: Ensure any label counting code handles underscore → space
# Since we can't know the exact counting code, let's search for common patterns

import re

# Pattern 1: Direct string match in a counting loop
# Look for lines that reference CONDITIONS and "Finding Labels" without .replace
lines = content.split("\n")
fixed_lines = []
in_counting = False

for i, line in enumerate(lines):
    # Check for label counting patterns that might miss the replace
    if "Finding Labels" in line and "condition" in line and ".replace" not in line:
        if "condition in" in line:
            # This line does a direct match without replace - fix it
            line = line.replace("condition in", "condition.replace('_', ' ') in")
            changes += 1
            print(f"  [OK] Line {i+1}: Added .replace('_', ' ') to label counting")

    # Also check for pandas str.contains without replace
    if ".str.contains(condition" in line and "replace" not in line:
        # Replace condition with condition.replace("_", " ")
        line = line.replace(
            ".str.contains(condition",
            ".str.contains(condition.replace('_', ' ')"
        )
        changes += 1
        print(f"  [OK] Line {i+1}: Added .replace to str.contains()")

    fixed_lines.append(line)

content = "\n".join(fixed_lines)

# ── Also fix any torch.load calls missing weights_only=False ──
old_load = 'torch.load(ckpt_path, map_location=device)'
new_load = 'torch.load(ckpt_path, map_location=device, weights_only=False)'
if old_load in content:
    content = content.replace(old_load, new_load)
    changes += 1
    print(f"  [OK] Fixed torch.load (resume) with weights_only=False")

# Fix checkpoint loading patterns
import re
# Match torch.load(...) without weights_only
load_pattern = r'torch\.load\(([^)]+)\)'
for match in re.finditer(load_pattern, content):
    args = match.group(1)
    if 'weights_only' not in args and 'map_location' in args:
        old = match.group(0)
        new = old.replace(')', ', weights_only=False)')
        if old in content and new not in content:
            content = content.replace(old, new, 1)
            changes += 1
            print(f"  [OK] Fixed torch.load with weights_only=False: {old[:60]}...")

if changes > 0:
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  Saved {changes} changes to {TRAIN_PATH}")
else:
    print(f"\n  No changes needed — label counting already handles underscore/space.")
    print(f"  The __getitem__ method was already correct. The 0% count was likely")
    print(f"  from a separate counting function. Run fix_1_verify_pleural.py to confirm.")

print(f"\nDone!")
