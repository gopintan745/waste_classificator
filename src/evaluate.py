#from typing import Literal
import torch
import optuna
from torch.utils.data import DataLoader
from src.train import fit
from src.dataset import WasteDataset
from src.models.custom_cnn import WasteClassifierCNN
from src.models.transfer_model import build_transfer_model
from src.transforms import train_transforms, test_transforms

def final_evaluation(model_type, db_root, data_root, num_classes, dataset,
                     epochs=50, device='cuda',):
    """Run final training using the best HPO hyperparameters, then evaluate on test set."""
    
    # ============================================
    # Load best hyperparameters from Optuna study
    # ============================================
    storage_path = f"sqlite:///{db_root}/optuna_study.db"
    study = optuna.load_study(
        storage=storage_path,
        study_name=f"{model_type}_study"
    )
    
    if not study.trials:
        raise ValueError(f"Study '{model_type}_study' has no trials. Run HPO first.")
    if study.best_value is None:
        raise ValueError(f"No completed trials in '{model_type}_study'.")
    
    best_params = study.best_params
    print(f"Best validation F1: {study.best_value:.4f}")
    print(f"Best hyperparameters: {best_params}")
    
    # ============================================
    # Build datasets with optimal image size
    # ============================================
    img_size = best_params['img_size']
    train_dataset = WasteDataset(data_root, train_transforms(img_size), "train")
    val_dataset   = WasteDataset(data_root, test_transforms(img_size),   "val")
    test_dataset  = WasteDataset(data_root, test_transforms(img_size),   "test")
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # ============================================
    # Build model with best hyperparameters
    # ============================================
    if model_type == "custom_cnn":
        model = WasteClassifierCNN(
            num_classes=num_classes,
            base_filters=best_params['base_filters'],   # fixed: was 'best_filters'
            dropout=best_params['dropout'],
        )
    elif model_type == "transfer":
        model = build_transfer_model(
            arch=best_params['arch'],
            num_classes=num_classes,
            dropout=best_params['dropout'],
            freeze_backbone=False,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'custom_cnn' or 'transfer'.")

    if dataset == 'trashnet':
        save_dir = f'/kaggle/working/waste_classificator/experiments/{model_type}_{dataset}_final'
    elif dataset == 'merged':
        save_dir = f'/kaggle/working/waste_classificator/experiments/{model_type}_{dataset}_final'
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'trashnet' or 'merged'.")
    
    # ============================================
    # Run final training with best hyperparameters
    # ============================================
    return fit(
        model=model,
        train_ds=train_dataset,
        val_ds=val_dataset,
        test_ds=test_dataset,
        num_classes=num_classes,
        batch_size=best_params['batch_size'],           # fixed: was 'batch_szie'
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        optimizer_name=best_params['optimizer'],         # fixed: was 'optimizer_name'
        scheduler_name=best_params['scheduler'],         # fixed: was 'scheduer_name'
        device=device,
        epochs=epochs,
        save_dir=f"{save_dir}",
        patience=20,    # slightly more lenient than default 7
    )


# ============================================
# Usage
# ============================================
if __name__ == "__main__":
    DB_ROOT    = "/kaggle/working/waste_classificator/experiments/custom_cnn"
    DATA_ROOT  = "/kaggle/working/waste_classificator/data/processed"
    
    # Run for custom CNN
    model, history, test_metrics = final_evaluation(
        model_type="custom_cnn",
        db_root=DB_ROOT,
        data_root=DATA_ROOT,
        num_classes=6,
        epochs=50,
    )
    print("\n=== Custom CNN Final Test Metrics ===")
    print(test_metrics)
