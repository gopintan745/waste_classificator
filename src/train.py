import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall,
)
from pathlib import Path
import matplotlib.pyplot as plt


def get_optimizer(name, params, lr, weight_decay):
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            out = model(imgs)
            loss = criterion(out, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    f1_metric  = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    prec_metric= MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    rec_metric = MulticlassRecall(num_classes=num_classes, average="macro").to(device)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        acc_metric.update(preds, labels)
        f1_metric.update(preds, labels)
        prec_metric.update(preds, labels)
        rec_metric.update(preds, labels)

    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "macro_acc":  acc_metric.compute().item(),
        "macro_f1":   f1_metric.compute().item(),
        "macro_prec": prec_metric.compute().item(),
        "macro_rec":  rec_metric.compute().item(),
    }


def fit(
    model, train_ds, val_ds, test_ds,
    num_classes, device="cuda",
    epochs=30, batch_size=32, lr=1e-3, weight_decay=1e-4,
    optimizer_name="adamw", scheduler_name="cosine", use_amp=True,
    save_dir="experiments/run",
    patience=7,
):
    os.makedirs(save_dir, exist_ok=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, weight_decay)

    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(train_loader))
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_f1": []}
    best_val_f1, bad_epochs = 0, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val = evaluate(model, val_loader, criterion, device, num_classes)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        history["val_f1"].append(val["macro_f1"])

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} val_f1={val['macro_f1']:.4f} | "
              f"{time.time()-t0:.1f}s")

        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            bad_epochs = 0
            torch.save({"model": model.state_dict(),
                        "val_f1": best_val_f1,
                        "epoch": epoch,
                        "classes": train_ds.classes}, f"{save_dir}/best_model.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best & evaluate on test set
    ckpt = torch.load(f"{save_dir}/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)

    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump({"test": test_metrics, "history": history,
                   "best_val_f1": best_val_f1}, f, indent=2)

    # Plot curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train acc"); ax[1].plot(history["val_acc"], label="val acc")
    ax[1].plot(history["val_f1"], label="val F1")
    ax[1].set_title("Accuracy / F1"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    plt.close()

    return model, history, test_metrics
