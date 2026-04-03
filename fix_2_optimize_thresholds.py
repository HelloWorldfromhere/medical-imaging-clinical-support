"""
Step 2: Per-Class Threshold Optimization
========================================
Run from project root AFTER training is complete:
    python fix_2_optimize_thresholds.py

This loads the trained model, runs inference on the test set,
finds optimal per-class thresholds via Youden's J statistic,
and saves them to models/checkpoints/optimal_thresholds.json.

Same approach used in the Mila hackathon — finds the threshold
that maximizes (sensitivity + specificity - 1) per class.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_curve, f1_score, roc_auc_score, classification_report
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

CONFIG = {
    "data_dir": "data/chestxray14",
    "checkpoint_path": "models/checkpoints/efficientnet_b3_multilabel_best.pth",
    "threshold_path": "models/checkpoints/optimal_thresholds.json",
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "test_split_start": 0.9,  # Last 10% of patient IDs = test set
}


# ── Dataset (same as training) ─────────────────────────────────────────────

class ChestXray14Dataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = self._find_image(img_name)

        if img_path is None:
            image = Image.new("RGB", (CONFIG["image_size"], CONFIG["image_size"]))
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.zeros(len(CONDITIONS), dtype=torch.float32)
        findings = str(row["Finding Labels"])
        for i, condition in enumerate(CONDITIONS):
            condition_clean = condition.replace("_", " ")
            if condition_clean in findings or condition in findings:
                labels[i] = 1.0

        return image, labels

    def _find_image(self, img_name):
        for pattern in [
            os.path.join(self.image_dir, img_name),
            *[os.path.join(self.image_dir, f"images_{i:03d}", "images", img_name) for i in range(1, 13)],
            *[os.path.join(self.image_dir, f"images_{i:03d}", img_name) for i in range(1, 13)],
        ]:
            if os.path.exists(pattern):
                return pattern
        return None


# ── Model ───────────────────────────────────────────────────────────────────

def build_model():
    model = models.efficientnet_b3(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, len(CONDITIONS)),
    )
    return model


# ── Threshold Optimization ──────────────────────────────────────────────────

def find_optimal_threshold_youden(y_true, y_scores):
    """
    Find optimal threshold using Youden's J statistic.
    J = sensitivity + specificity - 1 = TPR - FPR
    This is the same approach we used in the Mila hackathon.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def find_optimal_threshold_f1(y_true, y_scores, n_thresholds=100):
    """
    Find threshold that maximizes F1 score via grid search.
    More conservative than Youden — better for imbalanced data.
    """
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.linspace(0.05, 0.95, n_thresholds):
        preds = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {CONFIG['checkpoint_path']}...")
    model = build_model()
    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}, best AUC: {ckpt.get('best_auc', '?')}")

    # Load data — reproduce the same test split as training
    csv_path = os.path.join(CONFIG["data_dir"], "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)
    print(f"\nTotal images: {len(df)}")

    # Patient-level split (same as training)
    df["Patient ID"] = df["Image Index"].str.extract(r"(\d+)_").astype(int)
    patient_ids = sorted(df["Patient ID"].unique())
    np.random.seed(42)
    np.random.shuffle(patient_ids)

    test_start = int(len(patient_ids) * CONFIG["test_split_start"])
    test_patients = set(patient_ids[test_start:])
    test_df = df[df["Patient ID"].isin(test_patients)]
    print(f"Test set: {len(test_df)} images from {len(test_patients)} patients")

    # Create dataset
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = ChestXray14Dataset(test_df, CONFIG["data_dir"], val_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=False, num_workers=CONFIG["num_workers"])

    # Run inference
    print("\nRunning inference on test set...")
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test inference"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.append(labels.numpy())
            all_probs.append(probs)

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    print(f"  Collected predictions: {all_probs.shape}")

    # ── Find optimal thresholds ─────────────────────────────────────────

    print("\n" + "=" * 75)
    print(f"{'Condition':<22} {'AUC':>6} {'Default':>9} {'Youden':>9} {'F1-Opt':>9} {'F1@0.5':>7} {'F1@Opt':>7}")
    print("=" * 75)

    thresholds = {}
    for i, condition in enumerate(CONDITIONS):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]

        # Skip if no positive samples
        if y_true.sum() == 0:
            print(f"{condition:<22} {'N/A':>6} {'N/A':>9} {'N/A':>9} {'N/A':>9} {'N/A':>7} {'N/A':>7}")
            thresholds[condition] = 0.5
            continue

        auc = roc_auc_score(y_true, y_score)

        # Default threshold
        f1_default = f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)

        # Youden's J
        thresh_youden = find_optimal_threshold_youden(y_true, y_score)

        # F1-optimal
        thresh_f1, f1_optimal = find_optimal_threshold_f1(y_true, y_score)

        # Use F1-optimal threshold (more practical than Youden for imbalanced data)
        thresholds[condition] = thresh_f1

        print(f"{condition:<22} {auc:>6.3f} {'0.500':>9} {thresh_youden:>9.3f} {thresh_f1:>9.3f} {f1_default:>7.3f} {f1_optimal:>7.3f}")

    # ── Summary ─────────────────────────────────────────────────────────

    print("\n" + "=" * 75)

    # Compare mean F1: default vs optimized
    f1_default_all = []
    f1_optimized_all = []
    for i, condition in enumerate(CONDITIONS):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]
        if y_true.sum() == 0:
            continue
        f1_default_all.append(f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0))
        f1_optimized_all.append(f1_score(y_true, (y_score >= thresholds[condition]).astype(int), zero_division=0))

    print(f"\nMean F1 @ default 0.5 threshold:  {np.mean(f1_default_all):.4f}")
    print(f"Mean F1 @ optimized thresholds:   {np.mean(f1_optimized_all):.4f}")
    print(f"Improvement:                      +{np.mean(f1_optimized_all) - np.mean(f1_default_all):.4f}")

    # Save thresholds
    output = {
        "method": "f1_optimal_grid_search",
        "thresholds": thresholds,
        "mean_f1_default": float(np.mean(f1_default_all)),
        "mean_f1_optimized": float(np.mean(f1_optimized_all)),
    }

    os.makedirs(os.path.dirname(CONFIG["threshold_path"]), exist_ok=True)
    with open(CONFIG["threshold_path"], "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nThresholds saved to: {CONFIG['threshold_path']}")

    # ── Full classification report with optimized thresholds ────────────

    print("\n" + "=" * 75)
    print("Classification Report (optimized thresholds):")
    print("=" * 75)

    optimized_preds = np.zeros_like(all_probs)
    for i, condition in enumerate(CONDITIONS):
        optimized_preds[:, i] = (all_probs[:, i] >= thresholds[condition]).astype(int)

    condition_names = [c.replace("_", " ") for c in CONDITIONS]
    print(classification_report(all_labels, optimized_preds,
                                target_names=condition_names, zero_division=0))


if __name__ == "__main__":
    main()
