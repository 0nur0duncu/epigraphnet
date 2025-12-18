"""
EpiGraphNet Değerlendirme - Makaledeki Yöntem
5 kez farklı random seed ile tam eğitim + test

Makaledeki yöntem:
1. Her run'da farklı seed ile 80/10/10 böl
2. Her run'da model sıfırdan eğit (validation + early stopping)
3. Her run'da test et
4. 5 run'ın ortalamasını al
"""

import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, Any, List

from models import EpiGraphNet
from data import BonnEEGDataset
from data.dataset import load_bonn_dataset
from utils import calculate_all_metrics, get_confusion_matrix


def load_config(config_path: str) -> Dict[str, Any]:
    """Konfigürasyon dosyasını yükler."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_data_splits(
    signals: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, tuple]:
    """
    Veriyi stratified olarak train/val/test'e böler.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # İlk bölme: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        signals, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=labels
    )
    
    # İkinci bölme: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=random_seed,
        stratify=y_temp
    )
    
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }


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


def train_full_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    run_id: int
) -> nn.Module:
    """
    Modeli tam olarak eğitir (validation + early stopping).
    Makaledeki yöntemle aynı.
    """
    training_config = config["training"]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )
    
    # Learning rate scheduler (Tablo I: LinearLR, start_factor=0.1)
    scheduler_config = training_config.get("scheduler", {})
    scheduler = LinearLR(
        optimizer,
        start_factor=scheduler_config.get("start_factor", 0.1),
        total_iters=scheduler_config.get("total_iters", 10)
    )
    
    # Early stopping (makaledeki gibi)
    early_stop_config = training_config.get("early_stopping", {})
    patience = early_stop_config.get("patience", 10)
    min_delta = early_stop_config.get("min_delta", 0.001)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0
    
    num_epochs = training_config["num_epochs"]
    
    print(f"\n  Eğitim başlıyor (max {num_epochs} epoch, patience={patience})...")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Early stopping kontrolü
        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Her 10 epoch'ta bir yazdır
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: Train Loss={train_metrics['loss']:.4f}, "
                  f"Train Acc={train_metrics['accuracy']:.2f}%, "
                  f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.2f}%")
        
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch} (best: epoch {best_epoch})")
            break
    
    # En iyi modeli yükle
    if best_state:
        model.load_state_dict(best_state)
        print(f"    En iyi model yüklendi (epoch {best_epoch}, val_loss={best_val_loss:.4f})")
    
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """Test değerlendirmesi."""
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
    
    return {
        "metrics": metrics,
        "predictions": all_preds,
        "labels": all_labels
    }


def run_paper_evaluation(
    config: Dict[str, Any],
    num_runs: int = 5,
    base_seed: int = 42
) -> Dict[str, Any]:
    """
    Makaledeki değerlendirme yöntemi:
    5 kez farklı random seed ile tam eğitim + test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    print(f"Değerlendirme: {num_runs} run (her run'da farklı seed ile tam eğitim)")
    print("="*70)
    
    data_config = config["data"]
    training_config = config["training"]
    
    # Tüm veriyi yükle
    signals, labels = load_bonn_dataset(
        data_config["data_dir"],
        binary=data_config.get("binary_classification", False)
    )
    
    print(f"Toplam veri: {len(signals)} örnek")
    print(f"Sınıf dağılımı: {np.bincount(labels)}")
    print("="*70)
    
    all_results = []
    all_predictions = []
    all_true_labels = []
    
    for run in range(1, num_runs + 1):
        print(f"\n{'='*70}")
        print(f"RUN {run}/{num_runs} (seed={base_seed + run - 1})")
        print(f"{'='*70}")
        
        # Her run için farklı seed
        seed = base_seed + run - 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Veriyi böl (80/10/10 - stratified)
        splits = create_data_splits(
            signals, labels,
            train_ratio=data_config["train_ratio"],
            val_ratio=data_config["val_ratio"],
            random_seed=seed
        )
        
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]
        
        print(f"  Veri bölümü: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # DataLoader'lar oluştur
        batch_size = training_config["batch_size"]
        
        train_dataset = BonnEEGDataset(X_train, y_train)
        val_dataset = BonnEEGDataset(X_val, y_val)
        test_dataset = BonnEEGDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Model oluştur (her run'da sıfırdan)
        model = EpiGraphNet.from_config(config).to(device)
        
        # Tam eğitim (validation + early stopping)
        model = train_full_model(
            model, train_loader, val_loader, config, device, run
        )
        
        # Test et
        result = evaluate_model(model, test_loader, device)
        all_results.append(result["metrics"])
        
        # Tahminleri topla
        all_predictions.extend(result["predictions"])
        all_true_labels.extend(result["labels"])
        
        print(f"\n  Run {run} Test Sonuçları:")
        print(f"    Accuracy:  {result['metrics']['accuracy']:.2f}%")
        print(f"    Recall:    {result['metrics']['recall']:.2f}%")
        print(f"    Precision: {result['metrics']['precision']:.2f}%")
        print(f"    F1:        {result['metrics']['f1']:.2f}%")
    
    # Ortalama ve std hesapla
    summary = {}
    for metric_name in ["accuracy", "recall", "precision", "f1"]:
        values = [r[metric_name] for r in all_results]
        summary[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values
        }
    
    # Genel confusion matrix
    summary["confusion_matrix"] = get_confusion_matrix(all_true_labels, all_predictions)
    summary["run_results"] = all_results
    
    return summary


def print_summary(summary: Dict[str, Any], config: Dict[str, Any], num_runs: int):
    """Sonuç özetini yazdırır."""
    print("\n" + "="*70)
    print("MAKALE YÖNTEMİ İLE DEĞERLENDİRME SONUÇ ÖZETİ")
    print("="*70)
    
    # Model bilgileri
    model_config = config.get("model", {})
    graph_config = model_config.get("graph", {})
    
    thresholding = graph_config.get("thresholding", "value")
    sparsity = graph_config.get("sparsity", 25)
    
    th_name = "DE" if thresholding == "value" else "BE"
    print(f"Model: EpiGraphNet_{th_name} (a={sparsity})")
    print(f"Yöntem: {num_runs} run × (80/10/10 split + tam eğitim + test)")
    print("-"*70)
    
    # Her run sonucu
    print("\nRun Bazında Sonuçlar:")
    print(f"{'Run':<8} {'Accuracy':<12} {'Recall':<12} {'Precision':<12} {'F1':<12}")
    print("-"*56)
    
    for i, result in enumerate(summary["run_results"], 1):
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
    print("\nMAKALE İLE KARŞILAŞTIRMA (Tablo II - 5 sınıf):")
    print("-"*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1':<12}")
    print("-"*60)
    
    # Makale sonuçları
    paper_results = {
        "CNN-LSTM [16]": (77.70, 76.97),
        "1D-CNN-LSTM [17]": (79.50, 78.91),
        f"EpiGraphNet_{th_name}(a={sparsity})": (82.47 if sparsity == 25 else 80.60, 
                                                  81.33 if sparsity == 25 else 79.60),
    }
    
    for model_name, (acc, f1) in paper_results.items():
        print(f"{model_name:<25} {acc:>8.2f}%    {f1:>8.2f}%")
    
    print("-"*60)
    our_acc = summary["accuracy"]["mean"]
    our_f1 = summary["f1"]["mean"]
    paper_acc = 82.47 if sparsity == 25 else 80.60
    paper_f1 = 81.33 if sparsity == 25 else 79.60
    
    acc_diff = our_acc - paper_acc
    f1_diff = our_f1 - paper_f1
    acc_sign = "+" if acc_diff >= 0 else ""
    f1_sign = "+" if f1_diff >= 0 else ""
    
    print(f"{'Bizim Sonuç':<25} {our_acc:>8.2f}%    {our_f1:>8.2f}%")
    print(f"{'Fark':<25} {acc_sign}{acc_diff:>7.2f}%    {f1_sign}{f1_diff:>7.2f}%")
    print("="*70)
    
    # Confusion Matrix
    print("\nGenel Confusion Matrix (Tüm Run'lar):")
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
        description="EpiGraphNet Değerlendirme - Makaledeki Yöntem"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Konfigürasyon dosyası yolu"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Run sayısı (varsayılan: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (varsayılan: 42)"
    )
    args = parser.parse_args()
    
    # Konfigürasyon yükle
    config = load_config(args.config)
    
    # Makaledeki yöntemle değerlendirme yap
    summary = run_paper_evaluation(
        config,
        num_runs=args.num_runs,
        base_seed=args.seed
    )
    
    # Özeti yazdır
    print_summary(summary, config, args.num_runs)


if __name__ == "__main__":
    main()
