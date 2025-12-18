"""
EpiGraphNet Değerlendirme Scripti
Makaledeki 5 kez çalıştırıp ortalama alma yöntemi
"""

import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any, List

from models import EpiGraphNet
from data import create_data_loaders
from utils import calculate_all_metrics, get_confusion_matrix, MetricTracker
from utils.visualization import plot_confusion_matrix, plot_training_curves


def load_config(config_path: str) -> Dict[str, Any]:
    """Konfigürasyon dosyasını yükler."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Model değerlendirmesi.
    
    Returns:
        Metrikler ve confusion matrix
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        logits = model(batch_x)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch_y.cpu().tolist())
    
    # Metrikleri hesapla
    metrics = calculate_all_metrics(all_labels, all_preds)
    
    # Confusion matrix
    cm = get_confusion_matrix(all_labels, all_preds)
    
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels
    }


def run_multiple_evaluations(
    config: Dict[str, Any],
    num_runs: int = 5,
    checkpoint_path: str = None
) -> Dict[str, Any]:
    """
    Makaledeki gibi modeli birden fazla kez çalıştırıp ortalama alır.
    
    Args:
        config: Konfigürasyon
        num_runs: Çalıştırma sayısı
        checkpoint_path: Model checkpoint yolu
        
    Returns:
        Ortalama ve std metrikler
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    data_config = config["data"]
    training_config = config["training"]
    
    all_results = []
    
    for run in range(1, num_runs + 1):
        print(f"\n{'='*50}")
        print(f"Çalıştırma {run}/{num_runs}")
        print('='*50)
        
        # Her çalıştırmada farklı seed
        seed = data_config.get("random_seed", 42) + run - 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Veri yükle
        loaders = create_data_loaders(
            data_dir=data_config["data_dir"],
            batch_size=training_config["batch_size"],
            train_ratio=data_config["train_ratio"],
            val_ratio=data_config["val_ratio"],
            binary=data_config.get("binary_classification", False),
            random_seed=seed
        )
        
        # Model oluştur
        model = EpiGraphNet.from_config(config).to(device)
        
        # Checkpoint yükle (varsa)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Checkpoint yüklendi: {checkpoint_path}")
        
        # Değerlendir
        result = evaluate(model, loaders["test"], device)
        all_results.append(result["metrics"])
        
        # Sonuçları yazdır
        metrics = result["metrics"]
        print(f"Doğruluk: {metrics['accuracy']:.2f}%")
        print(f"Duyarlılık: {metrics['recall']:.2f}%")
        print(f"Kesinlik: {metrics['precision']:.2f}%")
        print(f"F1: {metrics['f1']:.2f}%")
    
    # Ortalama ve std hesapla
    summary = {}
    for metric_name in ["accuracy", "recall", "precision", "f1"]:
        values = [r[metric_name] for r in all_results]
        summary[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values
        }
    
    return summary


def print_summary(summary: Dict[str, Any], config: Dict[str, Any]):
    """Sonuç özetini yazdırır."""
    print("\n" + "="*60)
    print("SONUÇ ÖZETİ")
    print("="*60)
    
    # Model bilgileri
    model_config = config.get("model", {})
    graph_config = model_config.get("graph", {})
    
    thresholding = graph_config.get("thresholding", "value")
    sparsity = graph_config.get("sparsity", 50)
    
    th_name = "DE" if thresholding == "value" else "BE"
    print(f"Model: EpiGraphNet_{th_name} (a={sparsity})")
    print("-"*60)
    
    # Metrikler tablosu
    print(f"{'Metrik':<15} {'Ortalama':<15} {'Std':<15}")
    print("-"*45)
    
    for metric_name in ["accuracy", "recall", "precision", "f1"]:
        mean = summary[metric_name]["mean"]
        std = summary[metric_name]["std"]
        print(f"{metric_name.capitalize():<15} {mean:>8.2f}%      ±{std:>6.2f}%")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="EpiGraphNet Değerlendirme")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Konfigürasyon dosyası yolu"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint yolu"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Çalıştırma sayısı (ortalama için)"
    )
    parser.add_argument(
        "--plot-cm",
        action="store_true",
        help="Confusion matrix çiz"
    )
    args = parser.parse_args()
    
    # Konfigürasyon yükle
    config = load_config(args.config)
    
    # Değerlendirme yap
    summary = run_multiple_evaluations(
        config,
        num_runs=args.num_runs,
        checkpoint_path=args.checkpoint
    )
    
    # Özeti yazdır
    print_summary(summary, config)


if __name__ == "__main__":
    main()
