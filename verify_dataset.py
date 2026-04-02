"""
Verify ChestX-ray14 dataset is properly extracted and ready for training.
Run after download and extraction:
    python verify_dataset.py
"""

import os
import glob

DATA_DIR = "data/chestxray14"

print(f"Checking dataset in: {DATA_DIR}\n")

# Check if directory exists
if not os.path.exists(DATA_DIR):
    print(f"ERROR: {DATA_DIR} not found.")
    print("Download with: kaggle datasets download -d nih-chest-xrays/data -p data/chestxray14")
    exit(1)

# List top-level contents
print("Top-level contents:")
for item in sorted(os.listdir(DATA_DIR)):
    full = os.path.join(DATA_DIR, item)
    if os.path.isdir(full):
        count = len(os.listdir(full))
        print(f"  [DIR]  {item}/ ({count} items)")
    else:
        size_mb = os.path.getsize(full) / 1024 / 1024
        print(f"  [FILE] {item} ({size_mb:.1f} MB)")

# Check for CSV
csv_candidates = [
    os.path.join(DATA_DIR, "Data_Entry_2017.csv"),
    os.path.join(DATA_DIR, "Data_Entry_2017_v2020.csv"),
]
csv_found = None
for c in csv_candidates:
    if os.path.exists(c):
        csv_found = c
        break

if csv_found:
    import pandas as pd
    df = pd.read_csv(csv_found)
    print(f"\nCSV found: {csv_found}")
    print(f"  Total entries: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample image name: {df['Image Index'].iloc[0]}")
else:
    print("\nWARNING: No CSV labels file found!")

# Count images
image_patterns = [
    os.path.join(DATA_DIR, "*.png"),
    os.path.join(DATA_DIR, "images_*", "images", "*.png"),
    os.path.join(DATA_DIR, "images_*", "*.png"),
    os.path.join(DATA_DIR, "images", "*.png"),
]

total_images = 0
for pattern in image_patterns:
    found = glob.glob(pattern)
    if found:
        total_images += len(found)
        print(f"\nImages found: {len(found)} matching {pattern}")

if total_images == 0:
    # Check for zip files that need extraction
    zips = glob.glob(os.path.join(DATA_DIR, "*.zip"))
    if zips:
        print(f"\nFound {len(zips)} zip files that need extraction:")
        for z in zips:
            size_gb = os.path.getsize(z) / 1024**3
            print(f"  {os.path.basename(z)} ({size_gb:.1f} GB)")
        print("\nExtract with:")
        print(f"  cd {DATA_DIR}")
        print("  for %f in (*.zip) do (tar -xf %f)")
    else:
        print("\nWARNING: No images found! Check dataset extraction.")
else:
    print(f"\nTotal images found: {total_images}")

# Verify a sample image loads
if total_images > 0:
    try:
        from PIL import Image
        sample = glob.glob(image_patterns[0]) or glob.glob(image_patterns[1]) or glob.glob(image_patterns[2])
        if sample:
            img = Image.open(sample[0])
            print(f"\nSample image loads OK: {os.path.basename(sample[0])} ({img.size}, {img.mode})")
    except Exception as e:
        print(f"\nImage load test failed: {e}")

print(f"\n{'='*50}")
if csv_found and total_images > 0:
    print("READY FOR TRAINING")
    print(f"Run: python train_multilabel.py")
elif csv_found and total_images == 0:
    print("CSV found but images need extraction")
else:
    print("Dataset not ready — check above errors")
print(f"{'='*50}")
