import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from src.models.custom_cnn import WasteClassifierCNN
from src.models.transfer_model import build_transfer_model
from src.train import train_one_epoch, evaluate
from src.dataset import WasteDataset
from src.transforms import train_transforms,val_transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from src.train import get_optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR


def objective(trial, model_type, data_root, num_classes, device, max_epochs=20):
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
    scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "onecycle"])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    img_size = trial.suggest_categorical("img_size", [192, 224, 256])

    if model_type == "custom":
        base_filters = trial.suggest_categorical("base_filters", [16, 32, 48])
        model = WasteClassifierCNN(num_classes=num_classes, base_filters=base_filters, dropout=dropout)
    elif model_type == "transfer":
        arch = trial.suggest_categorical("arch", ["resnet50", "efficientnet_b0", "convnext_tiny"])
        model = build_transfer_model(arch=arch, num_classes=num_classes, dropout=dropout, freeze_backbone=False)
    else:
        return "No model type of that sort. Choose between custom or transfer"

    train_ds = WasteDataset(data_root, train_transforms(img_size), "train")
    val_ds   = WasteDataset(data_root, val_transforms(img_size),   "val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, weight_decay)
    
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_name == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=max_epochs * len(train_loader))
    else:
        scheduler = None
    
    scaler = torch.amp.GradScaler(device=device, enabled=True)
    for epoch in range(1, max_epochs + 1):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val = evaluate(model, val_loader, criterion, device, num_classes)
        if scheduler is not None:
            scheduler.step()
            trial.report(val["acc"], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return val["acc"]

def run_search(model_type, data_root, num_classes, n_trials=30, max_epochs=20):
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name=f"{model_type}_study",
        storage=f"sqlite:///experiments/{model_type}/optuna_study.db",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(t, model_type, data_root, num_classes, "cuda", max_epochs),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print("Best params:", study.best_params)
    print("Best F1:", study.best_value)
    return study
