import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, Any
import json
from datetime import datetime

from models.baselines import BaselineCNNLSTM, Baseline1DCNNLSTM
from models.epigraphnet import EpiGraphNet
from data import BonnEEGDataset
from data.dataset import load_bonn_dataset
from utils.training import train_one_epoch, validate

EXPERIMENTS = [
    ("CNN-LSTM", None, None),
    ("1D-CNN-LSTM", None, None),
    ("EpiGraphNet", 50, "value"),
    ("EpiGraphNet", 25, "value"),
    ("EpiGraphNet", 50, "connection"),
    ("EpiGraphNet", 25, "connection"),
]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_model(model_type: str, num_classes: int, sparsity: float = 50, thresholding: str = "value") -> nn.Module:
    dropout = 0.1
    
    if model_type == "CNN-LSTM":
        return BaselineCNNLSTM(
            in_channels=1,
            conv_channels=[16, 32, 64],
            kernel_sizes=[5, 5, 5],
            lstm_hidden=64,
            lstm_layers=2,
            num_classes=num_classes,
            dropout=dropout
        )
    
    elif model_type == "1D-CNN-LSTM":
        return Baseline1DCNNLSTM(
            in_channels=1,
            conv_channels=[32, 64, 128, 64],
            kernel_sizes=[7, 5, 5, 3],
            lstm_hidden=64,
            lstm_layers=2,
            num_classes=num_classes,
            dropout=dropout
        )
    
    elif model_type.startswith("EpiGraphNet"):
        return EpiGraphNet(
            in_channels=1,
            conv_channels=[16, 32, 64],
            lstm_hidden=64,
            lstm_layers=2,
            num_nodes=16,
            sparsity=sparsity,
            thresholding=thresholding,
            gcn_hidden=64,
            gcn_layers=3,
            num_classes=num_classes,
            dropout=dropout
        )
    
    raise ValueError(f"Bilinmeyen model türü: {model_type}")


def run_single_experiment(
    model_type: str,
    signals: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    run_id: int,
    sparsity: float = 50,
    thresholding: str = "value",
    binary: bool = False
) -> Dict[str, float]:
    seed = 42 + run_id * 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    num_classes = 2 if binary else 5
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        signals, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(BonnEEGDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BonnEEGDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(BonnEEGDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    model = create_model(model_type, num_classes, sparsity, thresholding).to(device)
    
    training_config = config["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"])
    
    num_epochs = training_config["num_epochs"]
    # if model_type.startswith("EpiGraphNet"):
    #     num_epochs = max(num_epochs, 100)
    
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    
    patience = 25 if model_type.startswith("EpiGraphNet") else 15
    min_delta = 0.001
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        if val_metrics["accuracy"] > best_val_acc + min_delta:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return validate(model, test_loader, criterion, device)


def run_experiment_suite(
    model_type: str,
    signals: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    num_runs: int = 5,
    sparsity: float = 50,
    thresholding: str = "value",
    binary: bool = False
) -> Dict[str, float]:
    all_results = []
    
    for run_id in range(1, num_runs + 1):
        metrics = run_single_experiment(
            model_type=model_type,
            signals=signals,
            labels=labels,
            config=config,
            device=device,
            run_id=run_id,
            sparsity=sparsity,
            thresholding=thresholding,
            binary=binary
        )
        all_results.append(metrics)
    
    avg_results = {}
    for key in all_results[0].keys():
        if key != "loss":
            values = [r[key] for r in all_results]
            avg_results[key] = np.mean(values)
            avg_results[f"{key}_std"] = np.std(values)
    
    return avg_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--multiclass-only", action="store_true")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classification_types = []
    if not args.multiclass_only:
        classification_types.append(("binary", True))
    if not args.binary_only:
        classification_types.append(("multiclass", False))
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "num_runs": args.num_runs,
        "experiments": {}
    }
    
    for class_name, binary in classification_types:
        signals, labels = load_bonn_dataset(config["data"]["data_dir"], binary=binary)
        all_results["experiments"][class_name] = {}
        
        for model_type, sparsity, thresholding in EXPERIMENTS:
            if model_type == "EpiGraphNet":
                th_str = "DE" if thresholding == "value" else "BE"
                model_name = f"EpiGraphNet_{th_str}(a={sparsity})"
            else:
                model_name = model_type
            
            results = run_experiment_suite(
                model_type=model_type,
                signals=signals,
                labels=labels,
                config=config,
                device=device,
                num_runs=args.num_runs,
                sparsity=sparsity if sparsity else 50,
                thresholding=thresholding if thresholding else "value",
                binary=binary
            )
            
            all_results["experiments"][class_name][model_name] = results
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
