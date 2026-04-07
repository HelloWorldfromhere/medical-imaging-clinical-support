# =============================================================
# models/visualizer.py
# Training Visualization & Evaluation Plots
#
# PURPOSE: Generate professional plots that document model
# training behavior and final performance. These go into
# the evaluation/ folder and are referenced in ARCHITECTURE.md
#
# WHY VISUALIZATION MATTERS IN PRODUCTION:
# - Loss curves reveal overfitting immediately
# - Confusion matrices show WHERE the model fails
# - Sample predictions build trust with non-technical stakeholders
# - All serious ML teams (Google, Meta, Mila) require these
#
# TOOLS USED:
# - Matplotlib: Industry standard plotting (seaborn for styling)
# - Seaborn: Built on matplotlib, better default aesthetics
# WHY NOT Plotly: Plotly is interactive (good for dashboards)
# but matplotlib is standard for saved research figures.
# =============================================================

import matplotlib

matplotlib.use('Agg')  # WHY: Non-interactive backend for saving files
                       # without needing a display (works on servers too)
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

# Set professional plot style
# WHY seaborn style: Cleaner than matplotlib defaults,
# closer to what you see in research papers
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150


def plot_training_curves(history: dict, model_name: str, save_dir: str):
    """
    Plot loss and accuracy curves over training epochs.

    WHY THIS IS CRITICAL:
    This is the single most important diagnostic plot in ML.
    It tells you:
    - Is the model learning? (loss should decrease)
    - Is it overfitting? (train acc >> val acc = overfit)
    - Did it converge? (curves flattening = good)
    - Should you train longer? (still improving = yes)

    WHAT TO LOOK FOR:
    - Healthy: Both curves decrease together, close to each other
    - Overfitting: Train loss drops, val loss increases → add dropout
    - Underfitting: Both losses stay high → increase model capacity
    - Learning rate too high: Loss oscillates wildly

    Args:
        history: dict with keys 'train_loss', 'val_loss',
                 'train_acc', 'val_acc' — each a list per epoch
        model_name: e.g. 'ResNet50'
        save_dir: folder to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} — Training Curves', fontsize=16, fontweight='bold')

    # ── Loss curves ──────────────────────────────────────────
    ax1.plot(epochs, history['train_loss'],
             'b-o', linewidth=2, markersize=4, label='Train Loss')
    ax1.plot(epochs, history['val_loss'],
             'r-o', linewidth=2, markersize=4, label='Val Loss')
    ax1.set_title('Loss over Epochs', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend(fontsize=11)
    ax1.set_xlim(1, len(epochs))

    # Highlight best val loss epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    ax1.axvline(x=best_epoch, color='green', linestyle='--',
                alpha=0.7, label=f'Best epoch: {best_epoch}')
    ax1.legend(fontsize=11)

    # ── Accuracy curves ───────────────────────────────────────
    ax2.plot(epochs, history['train_acc'],
             'b-o', linewidth=2, markersize=4, label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'],
             'r-o', linewidth=2, markersize=4, label='Val Accuracy')
    ax2.set_title('Accuracy over Epochs', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(fontsize=11)
    ax2.set_xlim(1, len(epochs))
    ax2.set_ylim(0, 105)

    # Highlight best val acc epoch
    best_acc_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    ax2.axvline(x=best_acc_epoch, color='green', linestyle='--', alpha=0.7)
    ax2.annotate(f'Best: {best_acc:.1f}%',
                 xy=(best_acc_epoch, best_acc),
                 xytext=(best_acc_epoch + 0.5, best_acc - 5),
                 fontsize=10, color='green')

    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name.lower().replace("-","_")}_training_curves.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path}")
    return path


def plot_confusion_matrix(y_true, y_pred, classes: list,
                          model_name: str, save_dir: str):
    """
    Plot confusion matrix for classification results.

    WHY CONFUSION MATRIX:
    Accuracy alone is MISLEADING for medical classification.
    Example: If 75% of images are PNEUMONIA, a model that
    always predicts PNEUMONIA gets 75% accuracy but is useless.

    The confusion matrix shows:
    - True Positives (correctly detected PNEUMONIA)
    - False Negatives (PNEUMONIA missed — most dangerous!)
    - False Positives (NORMAL flagged as PNEUMONIA)
    - True Negatives (correctly identified NORMAL)

    In medical AI, FALSE NEGATIVES (missing pneumonia) are
    FAR more dangerous than false positives. The confusion
    matrix makes this visible at a glance.

    INTERVIEW TIP: When asked "how did you evaluate your model?"
    mention this tradeoff explicitly. It shows clinical thinking.
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    # Also compute normalized version (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{model_name} — Confusion Matrix', fontsize=16, fontweight='bold')

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        ax=ax1, linewidths=0.5, linecolor='gray',
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax1.set_title('Raw Counts', fontsize=13)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)

    # Normalized percentages
    sns.heatmap(
        cm_normalized, annot=True, fmt='.1%', cmap='RdYlGn',
        xticklabels=classes, yticklabels=classes,
        ax=ax2, linewidths=0.5, linecolor='gray',
        annot_kws={"size": 13}
    )
    ax2.set_title('Normalized (% of true class)', fontsize=13)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)

    # Add interpretation note
    fig.text(0.5, 0.01,
             '⚠️  False Negatives (PNEUMONIA predicted as NORMAL) are clinically critical',
             ha='center', fontsize=10, color='darkred', style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(save_dir, f'{model_name.lower().replace("-","_")}_confusion_matrix.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path}")
    return path


def plot_model_comparison(results: dict, save_dir: str):
    """
    Side-by-side bar chart comparing ResNet50 vs EfficientNet-B3.

    WHY THIS PLOT:
    This is the "money shot" for your portfolio and interviews.
    It directly shows your evaluation methodology — you didn't
    just train one model and call it done. You compared
    architectures scientifically and documented which won and why.

    This plot goes in your README and ARCHITECTURE.md.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(results.keys())
    metrics_to_plot = ['test_acc', 'val_accuracy',
                       'pneumonia_recall', 'normal_f1']
    metric_labels = ['Test Accuracy (%)', 'Best Val Accuracy (%)',
                     'Pneumonia Recall (%)', 'Normal F1 (%)']

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle('Architecture Comparison: ResNet50 vs EfficientNet-B3',
                 fontsize=15, fontweight='bold')

    colors = ['#2196F3', '#FF5722']  # Blue for ResNet, Orange for EfficientNet

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        values = []
        for model in model_names:
            val = results[model].get(metric, 0)
            # Convert 0-1 scale to percentage if needed
            if val <= 1.0:
                val *= 100
            values.append(val)

        bars = ax.bar(model_names, values, color=colors,
                      edgecolor='black', linewidth=0.8, width=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

        # Highlight the winner
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.set_ylabel('Score (%)')
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path}")
    return path


def plot_sample_predictions(model, test_loader, classes: list,
                            model_name: str, device, save_dir: str,
                            num_samples: int = 16):
    """
    Grid of sample X-ray images with predicted vs true labels.

    WHY THIS PLOT:
    - Makes the model tangible and interpretable
    - Immediately reveals systematic errors (e.g. all wrong predictions
      are blurry images → data quality issue)
    - Essential for presentations and demos
    - Non-technical stakeholders understand this instantly

    Green border = correct prediction
    Red border   = wrong prediction (model error)
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    images_shown = 0
    sample_images = []
    sample_labels = []
    sample_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            _, preds = outputs.max(1)

            for i in range(len(images)):
                if images_shown >= num_samples:
                    break
                sample_images.append(images[i])
                sample_labels.append(labels[i].item())
                sample_preds.append(preds[i].item())
                images_shown += 1

            if images_shown >= num_samples:
                break

    # Plot grid
    cols = 4
    rows = num_samples // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(f'{model_name} — Sample Predictions on Test Set',
                 fontsize=15, fontweight='bold', y=1.01)

    # ImageNet denormalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for idx, ax in enumerate(axes.flatten()):
        if idx >= len(sample_images):
            ax.axis('off')
            continue

        # Denormalize image for display
        img = sample_images[idx].permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)

        true_label = classes[sample_labels[idx]]
        pred_label = classes[sample_preds[idx]]
        correct = sample_labels[idx] == sample_preds[idx]

        ax.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}",
            fontsize=9,
            color='green' if correct else 'red',
            fontweight='bold'
        )

        # Border color indicates correct/wrong
        for spine in ax.spines.values():
            spine.set_edgecolor('green' if correct else 'red')
            spine.set_linewidth(3)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(save_dir,
                        f'{model_name.lower().replace("-","_")}_sample_predictions.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path}")
    return path


def generate_all_plots(model, model_name, history,
                       y_true, y_pred, classes,
                       test_loader, device,
                       save_dir="evaluation/plots"):
    """
    Convenience function — generates all plots for one model.
    Call this at the end of training for each model.
    """
    print(f"\nGenerating evaluation plots for {model_name}...")

    plot_training_curves(history, model_name, save_dir)
    plot_confusion_matrix(y_true, y_pred, classes, model_name, save_dir)
    plot_sample_predictions(model, test_loader, classes,
                            model_name, device, save_dir)

    print(f"All plots saved to {save_dir}/")
