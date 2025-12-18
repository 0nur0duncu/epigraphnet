"""
EpiGraphNet Stratified K-Fold Cross-Validation Değerlendirme
Makaledeki sonuçlarla tam eşleşme için k-fold çapraz doğrulama
"""

import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

from models import EpiGraphNet
from data import BonnEEGDataset
from data.dataset import load_bonn_dataset
from utils import calculate_all_metrics, get_confusion_matrix


def load_config(config_path: str) -> Dict[str, Any]:
    """Konfigürasyon dosyasını yükler."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Bir epoch eğitim."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch_y.cpu().tolist())
    
    avg_loss = total_loss / len(train_loader.dataset)
    metrics = calculate_all_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validasyon."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch_y.cpu().tolist())
    
    avg_loss = total_loss / len(val_loader.dataset)
    metrics = calculate_all_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    
    return metrics


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    fold: int
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Tek bir fold için eğitim.
    
    Returns:
        (best_model, best_metrics)
    """
    training_config = config["training"]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )
    
    # Early stopping
    patience = training_config.get("early_stopping", {}).get("patience", 15)
    min_delta = training_config.get("early_stopping", {}).get("min_delta", 0.001)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_metrics = None
    
    num_epochs = training_config["num_epochs"]
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Early stopping kontrolü
        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Her 10 epoch'ta bir yazdır
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Train Loss={train_metrics['loss']:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.2f}%")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # En iyi modeli yükle
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_metrics


@torch.no_grad()
def evaluate_fold(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Fold değerlendirmesi."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        logits = model(batch_x)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch_y.cpu().tolist())
    
    metrics = calculate_all_metrics(all_labels, all_preds)
    return metrics


def run_kfold_cv(
    config: Dict[str, Any],
    n_splits: int = 5,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Stratified K-Fold Cross-Validation.
    
    Args:
        config: Konfigürasyon
        n_splits: Fold sayısı (k)
        random_seed: Rastgele tohum
        
    Returns:
        Her fold için metrikler ve ortalama/std
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    print(f"K-Fold: {n_splits} fold")
    print("="*60)
    
    data_config = config["data"]
    training_config = config["training"]
    
    # Tüm veriyi yükle
    signals, labels = load_bonn_dataset(
        data_config["data_dir"],
        binary=data_config.get("binary_classification", False)
    )
    
    print(f"Toplam veri: {len(signals)} örnek")
    print(f"Sınıf dağılımı: {np.bincount(labels)}")
    print("="*60)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    fold_results = []
    all_predictions = []
    all_true_labels = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(signals, labels), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*60}")
        print(f"Train: {len(train_idx)} örnek, Test: {len(test_idx)} örnek")
        
        # Seed ayarla (tekrarlanabilirlik için)
        torch.manual_seed(random_seed + fold)
        np.random.seed(random_seed + fold)
        
        # Train/Val split (train içinden %10 validation)
        train_signals = signals[train_idx]
        train_labels = labels[train_idx]
        test_signals = signals[test_idx]
        test_labels = labels[test_idx]
        
        # Validation için train'den ayır
        val_size = int(len(train_signals) * 0.125)  # ~%10 of total
        val_idx = np.random.choice(len(train_signals), val_size, replace=False)
        train_mask = np.ones(len(train_signals), dtype=bool)
        train_mask[val_idx] = False
        
        val_signals = train_signals[val_idx]
        val_labels = train_labels[val_idx]
        train_signals = train_signals[train_mask]
        train_labels = train_labels[train_mask]
        
        print(f"  → Train: {len(train_signals)}, Val: {len(val_signals)}, Test: {len(test_signals)}")
        
        # Dataset ve DataLoader oluştur
        train_dataset = BonnEEGDataset(train_signals, train_labels)
        val_dataset = BonnEEGDataset(val_signals, val_labels)
        test_dataset = BonnEEGDataset(test_signals, test_labels)
        
        batch_size = training_config["batch_size"]
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Model oluştur
        model = EpiGraphNet.from_config(config).to(device)
        
        # Eğit
        model, train_metrics = train_fold(
            model, train_loader, val_loader, config, device, fold
        )
        
        # Test et
        test_metrics = evaluate_fold(model, test_loader, device)
        fold_results.append(test_metrics)
        
        # Tüm tahminleri topla (confusion matrix için)
        model.eval()
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            all_predictions.extend(preds.cpu().tolist())
            all_true_labels.extend(batch_y.tolist())
        
        print(f"\n  Fold {fold} Sonuçları:")
        print(f"    Accuracy:  {test_metrics['accuracy']:.2f}%")
        print(f"    Recall:    {test_metrics['recall']:.2f}%")
        print(f"    Precision: {test_metrics['precision']:.2f}%")
        print(f"    F1:        {test_metrics['f1']:.2f}%")
    
    # Ortalama ve std hesapla
    summary = {}
    for metric_name in ["accuracy", "recall", "precision", "f1"]:
        values = [r[metric_name] for r in fold_results]
        summary[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values
        }
    
    # Genel confusion matrix
    summary["confusion_matrix"] = get_confusion_matrix(all_true_labels, all_predictions)
    summary["fold_results"] = fold_results
    
    return summary


def print_summary(summary: Dict[str, Any], config: Dict[str, Any], n_splits: int):
    """Sonuç özetini yazdırır."""
    print("\n" + "="*70)
    print("STRATIFIED K-FOLD CROSS-VALIDATION SONUÇ ÖZETİ")
    print("="*70)
    
    # Model bilgileri
    model_config = config.get("model", {})
    graph_config = model_config.get("graph", {})
    
    thresholding = graph_config.get("thresholding", "value")
    sparsity = graph_config.get("sparsity", 25)
    
    th_name = "DE" if thresholding == "value" else "BE"
    print(f"Model: EpiGraphNet_{th_name} (a={sparsity})")
    print(f"K-Fold: {n_splits}-fold Stratified Cross-Validation")
    print("-"*70)
    
    # Her fold sonucu
    print("\nFold Bazında Sonuçlar:")
    print(f"{'Fold':<8} {'Accuracy':<12} {'Recall':<12} {'Precision':<12} {'F1':<12}")
    print("-"*56)
    
    for i, result in enumerate(summary["fold_results"], 1):
        print(f"{i:<8} {result['accuracy']:>8.2f}%    {result['recall']:>8.2f}%    "
              f"{result['precision']:>8.2f}%    {result['f1']:>8.2f}%")
    
    print("-"*56)
    
    # Ortalama ve std
    print(f"\n{'Metrik':<15} {'Ortalama':<15} {'Std':<15}")
    print("-"*45)
    
    for metric_name in ["accuracy", "recall", "precision", "f1"]:
        mean = summary[metric_name]["mean"]
        std = summary[metric_name]["std"]
        print(f"{metric_name.capitalize():<15} {mean:>8.2f}%      ±{std:>6.2f}%")
    
    print("="*70)
    
    # Makale karşılaştırması
    print("\nMAKALE İLE KARŞILAŞTIRMA (Tablo II - 5 sınıf, a=25):")
    print("-"*50)
    print(f"{'Metrik':<12} {'Bizim Sonuç':<18} {'Makale':<15}")
    print("-"*50)
    
    paper_results = {
        "accuracy": 82.47,
        "recall": 82.47,  # Makale accuracy ile aynı
        "precision": 81.29,  # Tahmini
        "f1": 81.33
    }
    
    for metric in ["accuracy", "f1"]:
        our = summary[metric]["mean"]
        paper = paper_results[metric]
        diff = our - paper
        sign = "+" if diff >= 0 else ""
        print(f"{metric.capitalize():<12} {our:>8.2f}%          {paper:>8.2f}%       ({sign}{diff:.2f}%)")
    
    print("="*70)
    
    # Confusion Matrix
    print("\nGenel Confusion Matrix (Tüm Fold'lar):")
    cm = summary["confusion_matrix"]
    class_names = ["S", "N", "F", "O", "Z"]
    
    print(f"\n{'':>8}", end="")
    for name in class_names:
        print(f"{name:>6}", end="")
    print("  ← Tahmin")
    
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>6}  ", end="")
        for val in row:
            print(f"{val:>6}", end="")
        print()
    print("  ↑")
    print("Gerçek")


def main():
    parser = argparse.ArgumentParser(
        description="EpiGraphNet Stratified K-Fold Cross-Validation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Konfigürasyon dosyası yolu"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Fold sayısı (varsayılan: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (varsayılan: 42)"
    )
    args = parser.parse_args()
    
    # Konfigürasyon yükle
    config = load_config(args.config)
    
    # K-Fold Cross-Validation yap
    summary = run_kfold_cv(
        config,
        n_splits=args.k,
        random_seed=args.seed
    )
    
    # Özeti yazdır
    print_summary(summary, config, args.k)


if __name__ == "__main__":
    main()
