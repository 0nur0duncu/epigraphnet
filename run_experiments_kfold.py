"""
K-Fold Cross Validation ile Deneyler
Makaledeki sonuçlara daha yakın sonuçlar için stratified k-fold kullanılır.
"""

import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any, List
import json
from datetime import datetime
from tqdm import tqdm

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


def run_single_fold(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    fold_id: int,
    sparsity: float = 50,
    thresholding: str = "value",
    num_classes: int = 5,
    verbose: bool = True
) -> Dict[str, float]:
    
    batch_size = config["training"]["batch_size"]
    
    train_loader = DataLoader(BonnEEGDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BonnEEGDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(BonnEEGDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    model = create_model(model_type, num_classes, sparsity, thresholding).to(device)
    
    training_config = config["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"])
    
    num_epochs = training_config["num_epochs"]
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    
    patience = 20
    min_delta = 0.001
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    best_epoch = 0
    
    pbar = tqdm(range(1, num_epochs + 1), desc=f"  Fold {fold_id}", leave=False, ncols=100) if verbose else range(1, num_epochs + 1)
    
    for epoch in pbar:
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        if verbose and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'loss': f"{train_metrics['loss']:.4f}",
                'val_acc': f"{val_metrics['accuracy']:.2f}%",
                'best': f"{best_val_acc:.2f}%"
            })
        
        if val_metrics["accuracy"] > best_val_acc + min_delta:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                tqdm.write(f"    Early stopping at epoch {epoch} (best: {best_epoch})")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    if verbose:
        tqdm.write(f"    Fold {fold_id} - Test Acc: {test_metrics['accuracy']:.2f}%, F1: {test_metrics['f1']:.2f}%")
    
    return test_metrics


def run_kfold_experiment(
    model_type: str,
    signals: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    n_splits: int = 10,
    sparsity: float = 50,
    thresholding: str = "value",
    binary: bool = False,
    model_name: str = "",
    seed: int = 42
) -> Dict[str, float]:
    
    num_classes = 2 if binary else 5
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({n_splits}-Fold CV)")
    print(f"{'='*60}")
    
    # Outer K-fold for test
    outer_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    all_results = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_kfold.split(signals, labels), 1):
        X_train_val, X_test = signals[train_val_idx], signals[test_idx]
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]
        
        # Split train_val into train and validation (90/10 of train_val)
        inner_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed + fold_idx)
        train_idx, val_idx = next(inner_kfold.split(X_train_val, y_train_val))
        
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        # Set seed for reproducibility
        np.random.seed(seed + fold_idx)
        torch.manual_seed(seed + fold_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + fold_idx)
        
        metrics = run_single_fold(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            config=config,
            device=device,
            fold_id=fold_idx,
            sparsity=sparsity,
            thresholding=thresholding,
            num_classes=num_classes,
            verbose=True
        )
        all_results.append(metrics)
    
    # Calculate average results
    avg_results = {}
    for key in all_results[0].keys():
        if key != "loss":
            values = [r[key] for r in all_results]
            avg_results[key] = np.mean(values)
            avg_results[f"{key}_std"] = np.std(values)
    
    # Print summary
    print(f"\n  📊 {model_name} Ortalama Sonuçlar ({n_splits}-Fold CV):")
    print(f"     Accuracy:  {avg_results['accuracy']:.2f}% ± {avg_results['accuracy_std']:.2f}%")
    print(f"     Precision: {avg_results['precision']:.2f}% ± {avg_results['precision_std']:.2f}%")
    print(f"     Recall:    {avg_results['recall']:.2f}% ± {avg_results['recall_std']:.2f}%")
    print(f"     F1:        {avg_results['f1']:.2f}% ± {avg_results['f1_std']:.2f}%")
    
    return avg_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=10, help="Number of folds for cross-validation")
    parser.add_argument("--output", type=str, default="results_kfold.json")
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--multiclass-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    classification_types = []
    if not args.multiclass_only:
        classification_types.append(("binary", True))
    if not args.binary_only:
        classification_types.append(("multiclass", False))
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "n_splits": args.n_splits,
        "method": "stratified_k_fold_cv",
        "experiments": {}
    }
    
    for class_name, binary in classification_types:
        print(f"\n{'#'*70}")
        print(f"# Sınıflandırma Türü: {'İkili (Binary)' if binary else 'Çok Sınıflı (Multi-class)'}")
        print(f"# Yöntem: {args.n_splits}-Fold Stratified Cross Validation")
        print(f"{'#'*70}")
        
        signals, labels = load_bonn_dataset(config["data"]["data_dir"], binary=binary)
        print(f"Veri yüklendi: {len(signals)} örnek, {len(np.unique(labels))} sınıf")
        
        # Show class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Sınıf dağılımı: {dict(zip(unique, counts))}")
        
        all_results["experiments"][class_name] = {}
        
        for model_type, sparsity, thresholding in EXPERIMENTS:
            if model_type == "EpiGraphNet":
                th_str = "DE" if thresholding == "value" else "BE"
                model_name = f"EpiGraphNet_{th_str}(a={sparsity})"
            else:
                model_name = model_type
            
            results = run_kfold_experiment(
                model_type=model_type,
                signals=signals,
                labels=labels,
                config=config,
                device=device,
                n_splits=args.n_splits,
                sparsity=sparsity if sparsity else 50,
                thresholding=thresholding if thresholding else "value",
                binary=binary,
                model_name=model_name,
                seed=args.seed
            )
            
            all_results["experiments"][class_name][model_name] = results
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Tüm deneyler tamamlandı! Sonuçlar '{args.output}' dosyasına kaydedildi.")
    print(f"{'='*70}")
    
    # Print final comparison table
    print("\n" + "="*80)
    print("SONUÇ KARŞILAŞTIRMA TABLOSU")
    print("="*80)
    
    for class_name in all_results["experiments"]:
        print(f"\n{class_name.upper()}:")
        print("-"*70)
        print(f"{'Model':<30} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1':<15}")
        print("-"*70)
        for model_name, results in all_results["experiments"][class_name].items():
            acc = f"{results['accuracy']:.2f}±{results['accuracy_std']:.2f}"
            prec = f"{results['precision']:.2f}±{results['precision_std']:.2f}"
            rec = f"{results['recall']:.2f}±{results['recall_std']:.2f}"
            f1 = f"{results['f1']:.2f}±{results['f1_std']:.2f}"
            print(f"{model_name:<30} {acc:<15} {prec:<15} {rec:<15} {f1:<15}")


if __name__ == "__main__":
    main()
