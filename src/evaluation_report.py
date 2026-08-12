"""Generate per-class F1 table and confusion matrices for all models."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassF1Score, MulticlassPrecision, MulticlassRecall,
    MulticlassConfusionMatrix,
)
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================
# Replace these with your actual paths
PATHS = {
    "Custom CNN (TrashNet)": {
        "model_path": "/kaggle/working/waste_classificator/experiments/custom_cnn_trashnet_final/best_model.pth",
        "data_root":  "/kaggle/working/waste_classificator/data/processed",
        "arch":       "custom",
    },
    "Custom CNN (Merged)": {
        "model_path": "/kaggle/working/waste_classificator/experiments/custom_cnn_merged_final/best_model.pth",
        "data_root":  "/kaggle/input/datasets/godfredopintan/waste-classification",
        "arch":       "custom",
    },
    "EfficientNet-B0 (TrashNet)": {
        "model_path": "/kaggle/working/waste_classificator/experiments/transfer_trashnet_final/best_model.pth",
        "data_root":  "/kaggle/working/waste_classificator/data/processed",
        "arch":       "transfer",
    },
    "EfficientNet-B0 (Merged)": {
        "model_path": "/kaggle/working/waste_classificator/experiments/transfer_merged_final/best_model.pth",
        "data_root":  "/kaggle/input/datasets/godfredopintan/waste-classification",
        "arch":       "transfer",
    },
}

# Use the same class names as your training
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# ============================================
# LOAD MODEL
# ============================================
def load_model(model_path, data_root, arch, device="cuda"):
    """Load a trained model based on its architecture."""
    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # Inspect state_dict to determine architecture
    state_dict = ckpt["model"]
    num_classes = len(ckpt["classes"]) if "classes" in ckpt else len(CLASS_NAMES)
    
    if arch == "custom":
        from src.models.custom_cnn import WasteClassifierCNN
        # Use the default base_filters; if checkpoint was trained with 16,
        # you may need to pass it explicitly
        model = WasteClassifierCNN(num_classes=num_classes, base_filters=16, dropout=0.1)
    elif arch == "transfer":
        from src.models.transfer_model import build_transfer_model
        # Default to efficientnet_b0; adjust based on your saved model
        model = build_transfer_model("efficientnet_b0", num_classes=num_classes, 
                                      dropout=0.34, freeze_backbone=False)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


# ============================================
# PER-CLASS METRICS
# ============================================
def compute_per_class_metrics(model, loader, num_classes, class_names=CLASS_NAMES, device="cuda"):
    """Compute per-class precision, recall, F1, and confusion matrix."""
    f1_metric     = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    prec_metric   = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    rec_metric    = MulticlassRecall(num_classes=num_classes, average=None).to(device)
    cm_metric     = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    
    all_preds, all_labels = [], []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            all_preds.append(preds)
            all_labels.append(labels)
            f1_metric.update(preds, labels)
            prec_metric.update(preds, labels)
            rec_metric.update(preds, labels)
            cm_metric.update(preds, labels)
    
    f1   = f1_metric.compute().cpu().numpy()
    prec = prec_metric.compute().cpu().numpy()
    rec  = rec_metric.compute().cpu().numpy()
    cm   = cm_metric.compute().cpu().numpy()
    
    # Overall accuracy
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = (all_preds == all_labels).float().mean().item()
    
    return {
        "f1": f1,
        "prec": prec,
        "rec": rec,
        "cm": cm,
        "acc": acc,
    }


# ============================================
# PLOT CONFUSION MATRIX
# ============================================
def plot_confusion_matrix(cm, class_names, title, ax=None, normalize=True):
    """Plot a normalized confusion matrix."""
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # Replace 0/0 with 0
        fmt = ".2f"
        cmap = "Blues"
        vmax = 1.0
    else:
        cm_norm = cm
        fmt = "d"
        cmap = "Blues"
        vmax = cm.max()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.heatmap(
        cm_norm, annot=True, fmt=fmt, cmap=cmap, vmin=0, vmax=vmax,
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, ax=ax, square=True,
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    return ax


# ============================================
# PRINT PER-CLASS TABLE
# ============================================
def print_per_class_table(name, metrics, class_names):
    """Print a formatted per-class metrics table."""
    print(f"\n{'='*78}")
    print(f"  {name}")
    print(f"{'='*78}")
    print(f"  Overall Accuracy: {metrics['acc']:.4f}")
    print(f"  Macro F1:         {np.mean(metrics['f1']):.4f}")
    print()
    header = f"  {'Class':<12s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}"
    print(header)
    print(f"  {'-'*len(header)}")
    
    # Get support from CM row sums
    support = metrics['cm'].sum(axis=1)
    
    for i, cls in enumerate(class_names):
        print(f"  {cls:<12s} {metrics['prec'][i]:>10.4f} {metrics['rec'][i]:>10.4f} "
              f"{metrics['f1'][i]:>10.4f} {support[i]:>10.0f}")


# ============================================
# MAIN
# ============================================
def main():
    from src.dataset import WasteDataset
    from src.transforms import test_transforms
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Storage for all results
    all_metrics = {}
    
    # ============================================
    # FIGURE 1: 2x2 confusion matrices
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for i, (name, cfg) in enumerate(PATHS.items()):
        print(f"\n{'='*78}")
        print(f"Processing: {name}")
        print(f"{'='*78}")
        
        # Load model
        model = load_model(cfg["model_path"], cfg["data_root"], cfg["arch"], device)
        
        # Load test data
        test_ds = WasteDataset(cfg["data_root"], test_transforms(256), "test")
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
        
        # Compute metrics
        metrics = compute_per_class_metrics(
            model, test_loader, num_classes=len(CLASS_NAMES),
            class_names=CLASS_NAMES, device=device,
        )
        all_metrics[name] = metrics
        
        # Print table
        print_per_class_table(name, metrics, CLASS_NAMES)
        
        # Plot confusion matrix
        row, col = i // 2, i % 2
        plot_confusion_matrix(
            metrics["cm"], CLASS_NAMES,
            title=name,
            ax=axes[row, col],
        )
    
    plt.suptitle("Test Set Confusion Matrices", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✓ Saved confusion_matrices.png")
    
    # ============================================
    # FIGURE 2: Per-class F1 bar chart comparison
    # ============================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.2
    
    for i, (name, metrics) in enumerate(all_metrics.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, metrics["f1"], width, label=name, alpha=0.85)
        for bar, v in zip(bars, metrics["f1"]):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.2f}", ha='center', fontsize=7)
    
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Per-Class F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Score by Configuration", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved per_class_f1.png")
    
    # ============================================
    # FIGURE 3: Per-class precision-recall scatter
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    
    markers = ['o', 's', '^', 'D']
    colors = ['#4C72B0', '#DD8452', '#55A467', '#C44E52']
    
    for i, (name, metrics) in enumerate(all_metrics.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        for j, cls in enumerate(CLASS_NAMES):
            ax.scatter(metrics["rec"][j], metrics["prec"][j],
                       s=200, alpha=0.7,
                       marker=markers[j % len(markers)],
                       label=cls,
                       edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_title(name, fontsize=11)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
        
        # Add diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    
    plt.suptitle("Per-Class Precision vs Recall (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("precision_recall.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved precision_recall.png")
    
    # ============================================
    # CROSS-DATASET COMPARISON: TrashNet vs Merged
    # ============================================
    print("\n" + "="*78)
    print("  CROSS-DATASET COMPARISON")
    print("="*78)
    
    # Transfer learning: TrashNet vs Merged
    tl_trashnet = all_metrics["EfficientNet-B0 (TrashNet)"]["f1"]
    tl_merged   = all_metrics["EfficientNet-B0 (Merged)"]["f1"]
    
    print(f"\n{'Class':<12s} {'TrashNet F1':>14s} {'Merged F1':>14s} {'Δ (Merged-TrashNet)':>22s}")
    print("-" * 65)
    for i, cls in enumerate(CLASS_NAMES):
        delta = tl_merged[i] - tl_trashnet[i]
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {cls:<10s} {tl_trashnet[i]:>14.4f} {tl_merged[i]:>14.4f} "
              f"{delta:>+10.4f} {arrow}")
    
    # Custom CNN: TrashNet vs Merged
    print()
    cc_trashnet = all_metrics["Custom CNN (TrashNet)"]["f1"]
    cc_merged   = all_metrics["Custom CNN (Merged)"]["f1"]
    
    print(f"Custom CNN:")
    print(f"  {'Class':<10s} {'TrashNet F1':>14s} {'Merged F1':>14s} {'Δ (Merged-TrashNet)':>22s}")
    print("  " + "-" * 60)
    for i, cls in enumerate(CLASS_NAMES):
        delta = cc_merged[i] - cc_trashnet[i]
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {cls:<10s} {cc_trashnet[i]:>14.4f} {cc_merged[i]:>14.4f} "
              f"{delta:>+10.4f} {arrow}")
    
    # Save full report
    import json
    report = {}
    for name, metrics in all_metrics.items():
        report[name] = {
            "overall_accuracy": metrics["acc"],
            "macro_f1": float(np.mean(metrics["f1"])),
            "per_class": {
                cls: {
                    "precision": float(metrics["prec"][i]),
                    "recall": float(metrics["rec"][i]),
                    "f1": float(metrics["f1"][i]),
                    "support": int(metrics["cm"][i].sum()),
                }
                for i, cls in enumerate(CLASS_NAMES)
            },
            "confusion_matrix": metrics["cm"].tolist(),
        }
    
    with open("per_class_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n✓ Saved per_class_report.json")
    
    return all_metrics


if __name__ == "__main__":
    main()
