# =============================================================
# models/cnn_trainer.py
# Chest X-Ray CNN Training Pipeline
# Trains ResNet50 + EfficientNet-B3, compares, plots all results
# =============================================================

import json
import logging
import os
import sys
import time

import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visualizer import generate_all_plots, plot_model_comparison

load_dotenv()

CONFIG = {
    "data_dir":       "data/chest_xray/chest_xray",
    "checkpoint_dir": "models/checkpoints",
    "eval_dir":       "evaluation/plots",
    "batch_size":     32,
    "num_epochs":     15,
    "learning_rate":  0.001,
    "num_classes":    2,
    "image_size":     224,
    "num_workers":    0,
}

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "medical_rag"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

os.makedirs("logs", exist_ok=True)
os.makedirs(CONFIG["eval_dir"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def get_data_loaders():
    train_transform, val_transform = get_transforms()
    train_ds = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "val"),   transform=val_transform)
    test_ds  = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "test"),  transform=val_transform)

    logger.info(f"Class mapping : {train_ds.class_to_idx}")
    logger.info(f"Train samples : {len(train_ds)}")
    logger.info(f"Val samples   : {len(val_ds)} (note: tiny val set is known Kaggle quirk)")
    logger.info(f"Test samples  : {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes


def build_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model


def build_efficientnet_b3(num_classes):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    for param in model.features[:6].parameters():
        param.requires_grad = False
    for param in model.features[6:].parameters():
        param.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Eval ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / len(loader), 100.0 * correct / total, all_preds, all_labels


def log_model_to_db(model_type, version, accuracy, metrics, file_path):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("UPDATE model_versions SET is_active = FALSE WHERE model_type = %s", (model_type,))
        cur.execute("""
            INSERT INTO model_versions (model_type, version, accuracy, metrics, file_path, is_active)
            VALUES (%s, %s, %s, %s, %s, TRUE)
        """, (model_type, version, accuracy, json.dumps(metrics), file_path))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"  Logged to DB: {model_type} {version} ({accuracy:.2f}%)")
    except Exception as e:
        logger.error(f"  DB logging failed: {e}")


def train_model(model_name, model, train_loader, val_loader, test_loader, classes, device):
    logger.info(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"], weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    best_val_acc, best_model_path = 0.0, None

    # Track per-epoch metrics for loss/accuracy curve plots
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(CONFIG["num_epochs"]):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch [{epoch+1:02d}/{CONFIG['num_epochs']}] "
            f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% "
            f"| {time.time()-t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = os.path.join(CONFIG["checkpoint_dir"],
                                f"{model_name.lower().replace('-','_')}_best.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc, "val_loss": val_loss,
                "classes": classes, "model_name": model_name, "config": CONFIG,
            }, path)
            best_model_path = path
            logger.info(f"  ✅ Best checkpoint saved → val_acc={val_acc:.2f}%")

    # Final test evaluation
    logger.info("\n  Loading best checkpoint for test set evaluation...")
    ckpt = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    report = classification_report(test_labels, test_preds, target_names=classes, output_dict=True)

    logger.info(f"\n  ── {model_name} Final Results ──────────────────")
    logger.info(f"  Test Accuracy   : {test_acc:.2f}%")
    logger.info(f"  Best Val Acc    : {best_val_acc:.2f}%")
    logger.info(f"  NORMAL    → P: {report['NORMAL']['precision']:.3f} | R: {report['NORMAL']['recall']:.3f} | F1: {report['NORMAL']['f1-score']:.3f}")
    logger.info(f"  PNEUMONIA → P: {report['PNEUMONIA']['precision']:.3f} | R: {report['PNEUMONIA']['recall']:.3f} | F1: {report['PNEUMONIA']['f1-score']:.3f}")

    # Generate all evaluation plots
    logger.info("\n  Generating evaluation plots...")
    generate_all_plots(
        model=model, model_name=model_name, history=history,
        y_true=test_labels, y_pred=test_preds, classes=classes,
        test_loader=test_loader, device=device, save_dir=CONFIG["eval_dir"]
    )

    metrics = {
        "test_accuracy": test_acc, "val_accuracy": best_val_acc,
        "normal_f1": report['NORMAL']['f1-score'],
        "pneumonia_f1": report['PNEUMONIA']['f1-score'],
        "normal_recall": report['NORMAL']['recall'],
        "pneumonia_recall": report['PNEUMONIA']['recall'],
        "normal_precision": report['NORMAL']['precision'],
        "pneumonia_precision": report['PNEUMONIA']['precision'],
    }
    log_model_to_db(model_name, "v1.0", test_acc, metrics, best_model_path)
    return test_acc, best_val_acc, metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    if device.type == "cuda":
        logger.info(f"GPU    : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger.info("\nLoading datasets...")
    train_loader, val_loader, test_loader, classes = get_data_loaders()
    logger.info(f"Classes: {classes}")

    all_results = {}

    # Model 1: ResNet50
    logger.info("\n" + "="*60)
    logger.info("MODEL 1 OF 2: ResNet50 (Baseline)")
    logger.info("="*60)
    resnet = build_resnet50(CONFIG["num_classes"])
    acc, val_acc, metrics = train_model("ResNet50", resnet, train_loader, val_loader, test_loader, classes, device)
    all_results["ResNet50"] = {**metrics, "test_acc": acc}
    del resnet
    torch.cuda.empty_cache()

    # Model 2: EfficientNet-B3
    logger.info("\n" + "="*60)
    logger.info("MODEL 2 OF 2: EfficientNet-B3 (Challenger)")
    logger.info("="*60)
    effnet = build_efficientnet_b3(CONFIG["num_classes"])
    acc, val_acc, metrics = train_model("EfficientNet-B3", effnet, train_loader, val_loader, test_loader, classes, device)
    all_results["EfficientNet-B3"] = {**metrics, "test_acc": acc}

    # Comparison plot
    logger.info("\nGenerating architecture comparison plot...")
    plot_model_comparison(all_results, save_dir=CONFIG["eval_dir"])

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("ARCHITECTURE COMPARISON SUMMARY")
    logger.info("="*60)
    for name, r in all_results.items():
        logger.info(f"{name:20s} | Test Acc: {r['test_acc']:.2f}% | Pneumonia Recall: {r['pneumonia_recall']*100:.2f}%")

    winner = max(all_results, key=lambda x: all_results[x]["test_acc"])
    logger.info(f"\n✅ Winner: {winner} ({all_results[winner]['test_acc']:.2f}%)")
    logger.info(f"📊 Plots saved to: {CONFIG['eval_dir']}/")
    logger.info("📝 Next: document findings in ARCHITECTURE.md")
