import torch
import optuna
from torch.utils.data import DataLoader
from src.train import fit
from src.dataset import WasteDataset
from src.models.custom_cnn import WasteClassifierCNN
from src.models.transfer_model import build_transfer_model
from src.transforms import train_transforms, test_transforms

PROJECT_ROOT = '/kaggle/working/waste_classificator'

def final_evaluation(model_type, data_root, num_classes, device):
    study = optuna.load_study(
        storage = f"sqlite:///{PROJECT_ROOT}/experiments/{model_type}/optuna_study.db",
        study_name = f"{model_type}_study"
    )
    best_params = study.best_params

    train_dataset = WasteDataset(data_root, train_transforms(best_params['img_size']), "train")
    val_dataset = WasteDataset(data_root, test_transforms(best_params['img_size']), "val")
    test_dataset = WasteDataset(data_root, test_transforms(best_params['img_size']), "test")
    
    if model_type == "custom_cnn":
        model = WasteClassifierCNN(num_classes, base_filters=best_params['best_filters'], dropout=best_params['dropout'])
    elif model_type == "transfer":
        model = build_transfer_model(arch=best_params['arch'], num_classes=num_classes, dropout=best_params['dropout'], freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose between 'custom_cnn' or 'transfer'.")

    return fit(
        model=model,
        train_ds=train_dataset,
        val_ds = val_dataset,
        test_ds=test_dataset,
        num_classes=num_classes,
        batch_size=best_params['batch_szie'],
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        optimizer_name=best_params['optimizer_name'],
        scheduler_name=best_params['scheduer_name'],
        device=device
    )