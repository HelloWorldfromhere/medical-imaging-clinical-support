# =============================================================
# models/multilabel_visualizer.py
# Multi-Label Training Visualization & Evaluation Plots
#
# Generates professional plots for the 14-condition classifier:
# 1. Training curves (loss + AUC)
# 2. Per-class AUC-ROC bar chart
# 3. ROC curves overlay (all 14 conditions)
# 4. Per-class confusion matrices (2x7 grid)
# 5. Sample predictions with confidence bars
# =============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural Thick.",
    "Pneumonia", "Pneumothorax",
]


def plot_training_curves(history, save_dir):
    """Training and validation loss + AUC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], 'o-', label="Train Loss", color="#2c5364", linewidth=2, markersize=4)
    axes[0].plot(epochs, history["val_loss"], 's-', label="Val Loss", color="#e74c3c", linewidth=2, markersize=4)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("BCEWithLogits Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_auc"], 'o-', label="Train AUC", color="#2c5364", linewidth=2, markersize=4)
    axes[1].plot(epochs, history["val_auc"], 's-', label="Val AUC", color="#e74c3c", linewidth=2, markersize=4)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Mean AUC-ROC", fontsize=12)
    axes[1].set_title("Training & Validation AUC", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.0)

    plt.tight_layout()
    path = os.path.join(save_dir, "multilabel_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_per_class_auc(aucs, save_dir):
    """Horizontal bar chart of per-class AUC-ROC scores."""
    fig, ax = plt.subplots(figsize=(10, 7))

    conditions = list(aucs.keys())
    scores = [aucs[c] for c in conditions]
    sorted_pairs = sorted(zip(conditions, scores), key=lambda x: x[1], reverse=True)
    conditions = [p[0] for p in sorted_pairs]
    scores = [p[1] for p in sorted_pairs]

    colors = ["#065f46" if s >= 0.80 else "#854d0e" if s >= 0.70 else "#991b1b" for s in scores]

    bars = ax.barh(range(len(conditions)), scores, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=11)
    ax.set_xlabel("AUC-ROC", fontsize=12)
    ax.set_title("Per-Class AUC-ROC — ChestX-ray14 EfficientNet-B3", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.80, color="#065f46", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=0.70, color="#854d0e", linestyle="--", alpha=0.4, linewidth=1)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#065f46", label="Good (>0.80)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#854d0e", label="Fair (0.70-0.80)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#991b1b", label="Needs improvement (<0.70)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "multilabel_per_class_auc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(labels, preds, save_dir):
    """Overlay ROC curves for all 14 conditions."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set3(np.linspace(0, 1, 14))
    mean_fpr = np.linspace(0, 1, 100)
    all_tprs = []

    for i, condition in enumerate(CONDITIONS):
        if labels[:, i].sum() == 0 or labels[:, i].sum() == len(labels):
            continue
        fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], linewidth=1.5, alpha=0.8,
                label=f"{condition} ({roc_auc:.3f})")
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        all_tprs.append(interp_tpr)

    if all_tprs:
        mean_tpr = np.mean(all_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, 'k-', linewidth=2.5,
                label=f"Mean ROC ({mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All 14 Conditions", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    path = os.path.join(save_dir, "multilabel_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(labels, preds, save_dir, threshold=0.5):
    """2x7 grid of per-class confusion matrices."""
    binary_preds = (preds > threshold).astype(int)

    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    fig.suptitle("Per-Class Confusion Matrices — ChestX-ray14", fontsize=16, fontweight="bold", y=1.02)

    for i, condition in enumerate(CONDITIONS):
        row, col = i // 7, i % 7
        ax = axes[row, col]

        cm = confusion_matrix(labels[:, i], binary_preds[:, i], labels=[0, 1])
        if cm.shape == (2, 2):
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
                        cbar=False, annot_kws={"size": 10})
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)

        ax.set_title(condition, fontsize=10, fontweight="bold")
        ax.set_xlabel("" if row == 0 else "Predicted", fontsize=8)
        ax.set_ylabel("Actual" if col == 0 else "", fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "multilabel_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_label_distribution(labels, save_dir):
    """Bar chart showing label frequency in the dataset."""
    fig, ax = plt.subplots(figsize=(12, 5))

    counts = labels.sum(axis=0)
    sorted_idx = np.argsort(counts)[::-1]
    sorted_conditions = [CONDITIONS[i] for i in sorted_idx]
    sorted_counts = counts[sorted_idx]

    bars = ax.bar(range(len(sorted_conditions)), sorted_counts, color="#2c5364", edgecolor="white")
    ax.set_xticks(range(len(sorted_conditions)))
    ax.set_xticklabels(sorted_conditions, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Number of positive samples", fontsize=12)
    ax.set_title("Label Distribution — ChestX-ray14", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, count in zip(bars, sorted_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{int(count):,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "multilabel_label_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_all_multilabel_plots(history, test_labels, test_preds, test_aucs, save_dir):
    """Generate all visualization plots after training."""
    os.makedirs(save_dir, exist_ok=True)
    print("\nGenerating evaluation plots...")

    plot_training_curves(history, save_dir)
    plot_per_class_auc(test_aucs, save_dir)
    plot_roc_curves(test_labels, test_preds, save_dir)
    plot_confusion_matrices(test_labels, test_preds, save_dir)
    plot_label_distribution(test_labels, save_dir)

    print(f"\nAll plots saved to: {save_dir}/")
