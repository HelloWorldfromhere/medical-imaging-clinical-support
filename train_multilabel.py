"""
ChestX-ray14 Multi-Label CNN Training Pipeline
===============================================
Trains EfficientNet-B3 for 14-condition chest X-ray classification.

Dataset: NIH ChestX-ray14 (112,120 images, 14 pathology labels)
Architecture: EfficientNet-B3 with multi-label sigmoid output
Loss: BCEWithLogitsLoss (multi-label)
Metrics: Per-class AUC-ROC, per-class F1, mean AUC

Usage:
    python train_multilabel.py                    # Full training
    python train_multilabel.py --epochs 5         # Quick test
    python train_multilabel.py --resume           # Resume from checkpoint
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.multilabel_visualizer import generate_all_multilabel_plots

load_dotenv()

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]
NUM_CLASSES = len(CONDITIONS)

CONFIG = {
    "data_dir": "data/chestxray14",
    "checkpoint_dir": "models/checkpoints",
    "eval_dir": "evaluation/plots",
    "results_dir": "evaluation",
    "batch_size": 16,
    "num_epochs": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "image_size": 224,
    "num_workers": 4,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "patience": 5,
    "min_delta": 0.001,
}

os.makedirs("logs", exist_ok=True)
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["eval_dir"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/multilabel_training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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

        labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)
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


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"] + 32, CONFIG["image_size"] + 32)),
        transforms.RandomCrop(CONFIG["image_size"]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_model():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    for param in model.features[:6].parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, NUM_CLASSES),
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset), np.concatenate(all_preds), np.concatenate(all_labels)


def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]  ", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return running_loss / len(loader.dataset), np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(preds, labels):
    aucs, f1s = {}, {}
    binary_preds = (preds > 0.5).astype(int)

    for i, condition in enumerate(CONDITIONS):
        try:
            if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels):
                aucs[condition] = roc_auc_score(labels[:, i], preds[:, i])
            else:
                aucs[condition] = 0.0
        except ValueError:
            aucs[condition] = 0.0
        f1s[condition] = f1_score(labels[:, i], binary_preds[:, i], zero_division=0)

    mean_auc = np.mean([v for v in aucs.values() if v > 0])
    mean_f1 = np.mean(list(f1s.values()))
    return aucs, f1s, mean_auc, mean_f1


def load_data():
    data_dir = CONFIG["data_dir"]

    csv_path = None
    for candidate in [
        os.path.join(data_dir, "Data_Entry_2017.csv"),
        os.path.join(data_dir, "Data_Entry_2017_v2020.csv"),
    ]:
        if os.path.exists(candidate):
            csv_path = candidate
            break

    if csv_path is None:
        raise FileNotFoundError(f"CSV labels file not found in {data_dir}")

    logger.info(f"Loading labels from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Total images in CSV: {len(df)}")

    has_finding = df[df["Finding Labels"] != "No Finding"]
    no_finding = df[df["Finding Labels"] == "No Finding"].sample(
        n=min(len(has_finding), len(df[df["Finding Labels"] == "No Finding"])),
        random_state=42,
    )
    df = pd.concat([has_finding, no_finding]).reset_index(drop=True)
    logger.info(f"Filtered dataset: {len(df)} images")

    logger.info("Label distribution:")
    for condition in CONDITIONS:
        condition_clean = condition.replace("_", " ")
        count = df["Finding Labels"].str.contains(condition_clean).sum()
        logger.info(f"  {condition}: {count} ({100*count/len(df):.1f}%)")

    if "Patient ID" in df.columns:
        patient_ids = df["Patient ID"].unique()
        np.random.seed(42)
        np.random.shuffle(patient_ids)
        n_train = int(len(patient_ids) * CONFIG["train_split"])
        n_val = int(len(patient_ids) * CONFIG["val_split"])
        train_patients = set(patient_ids[:n_train])
        val_patients = set(patient_ids[n_train:n_train + n_val])
        train_df = df[df["Patient ID"].isin(train_patients)]
        val_df = df[df["Patient ID"].isin(val_patients)]
        test_df = df[~df["Patient ID"].isin(train_patients | val_patients)]
        logger.info(f"Patient-level split: {len(train_patients)} train, {len(val_patients)} val patients")
    else:
        n = len(df)
        indices = np.random.RandomState(42).permutation(n)
        n_train, n_val = int(n * 0.8), int(n * 0.1)
        train_df = df.iloc[indices[:n_train]]
        val_df = df.iloc[indices[n_train:n_train + n_val]]
        test_df = df.iloc[indices[n_train + n_val:]]

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df, val_df, test_df = load_data()
    train_transform, val_transform = get_transforms()
    image_dir = CONFIG["data_dir"]

    train_loader = DataLoader(
        ChestXray14Dataset(train_df, image_dir, train_transform),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(
        ChestXray14Dataset(val_df, image_dir, val_transform),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader = DataLoader(
        ChestXray14Dataset(test_df, image_dir, val_transform),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    model = build_model().to(device)
    logger.info(f"Model: EfficientNet-B3 ({sum(p.numel() for p in model.parameters()):,} params)")
    logger.info(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    start_epoch, best_auc = 0, 0.0
    if args.resume:
        ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "efficientnet_b3_multilabel_best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_auc = ckpt.get("best_auc", 0.0)
            logger.info(f"Resumed from epoch {start_epoch}, best AUC: {best_auc:.4f}")

    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    patience_counter = 0
    num_epochs = args.epochs or CONFIG["num_epochs"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {num_epochs} epochs, batch_size={CONFIG['batch_size']}")
    logger.info(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        train_loss, train_preds, train_labels = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_aucs, train_f1s, train_mean_auc, train_mean_f1 = compute_metrics(train_preds, train_labels)

        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device, epoch)
        val_aucs, val_f1s, val_mean_auc, val_mean_f1 = compute_metrics(val_preds, val_labels)

        scheduler.step(val_mean_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_mean_auc)
        history["val_auc"].append(val_mean_auc)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.0f}s) — "
            f"Train Loss: {train_loss:.4f}, AUC: {train_mean_auc:.4f} | "
            f"Val Loss: {val_loss:.4f}, AUC: {val_mean_auc:.4f}, F1: {val_mean_f1:.4f}"
        )

        if val_mean_auc > best_auc + CONFIG["min_delta"]:
            best_auc = val_mean_auc
            patience_counter = 0
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "efficientnet_b3_multilabel_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "val_aucs": val_aucs,
                "val_f1s": val_f1s,
                "conditions": CONDITIONS,
                "config": CONFIG,
            }, ckpt_path)
            logger.info(f"  *** New best model saved (AUC: {best_auc:.4f}) ***")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # ── Final Evaluation ────────────────────────────────────────────────────

    logger.info(f"\n{'='*60}")
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info(f"{'='*60}\n")

    ckpt = torch.load(
        os.path.join(CONFIG["checkpoint_dir"], "efficientnet_b3_multilabel_best.pth"),
        map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device, 0)
    test_aucs, test_f1s, test_mean_auc, test_mean_f1 = compute_metrics(test_preds, test_labels)

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Mean AUC: {test_mean_auc:.4f}")
    logger.info(f"Test Mean F1:  {test_mean_f1:.4f}")
    logger.info(f"\n{'Condition':<22} {'AUC':>8} {'F1':>8}")
    logger.info("-" * 40)
    for condition in CONDITIONS:
        logger.info(f"{condition:<22} {test_aucs[condition]:>8.4f} {test_f1s[condition]:>8.4f}")

    # ── Save Results ────────────────────────────────────────────────────────

    results = {
        "test_loss": test_loss,
        "test_mean_auc": test_mean_auc,
        "test_mean_f1": test_mean_f1,
        "per_class_auc": test_aucs,
        "per_class_f1": test_f1s,
        "conditions": CONDITIONS,
        "config": CONFIG,
        "history": history,
        "best_epoch": ckpt["epoch"] + 1,
    }
    with open(os.path.join(CONFIG["results_dir"], "multilabel_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ── Generate All Plots ──────────────────────────────────────────────────

    generate_all_multilabel_plots(
        history=history,
        test_labels=test_labels,
        test_preds=test_preds,
        test_aucs=test_aucs,
        save_dir=CONFIG["eval_dir"],
    )

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best model: {CONFIG['checkpoint_dir']}/efficientnet_b3_multilabel_best.pth")
    logger.info(f"Best val AUC: {best_auc:.4f} | Test AUC: {test_mean_auc:.4f}")
    logger.info(f"Plots saved to: {CONFIG['eval_dir']}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())
