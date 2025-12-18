"""
EpiGraphNet Eğitim Scripti
Makaledeki Tablo I hiperparametreleri ile Bonn EEG veri kümesi üzerinde eğitim
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from typing import Dict, Any, Optional

from models import EpiGraphNet
from data import create_data_loaders, download_bonn_dataset, verify_dataset
from utils import calculate_all_metrics, MetricTracker


def load_config(config_path: str) -> Dict[str, Any]:
    """Konfigürasyon dosyasını yükler."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_data_exists(data_dir: str) -> bool:
    """
    Veri kümesinin mevcut olduğundan emin olur.
    Yoksa indirir veya sentetik veri oluşturur.
    
    Args:
        data_dir: Veri dizini
        
    Returns:
        Başarılı mı
    """
    stats = verify_dataset(data_dir)
    
    if 'error' in stats or not stats.get('valid', False):
        print(f"\nVeri kümesi bulunamadı veya eksik: {data_dir}")
        print("Veri kümesi hazırlanıyor...")
        return download_bonn_dataset(data_dir)
    
    print(f"✓ Veri kümesi mevcut: {stats['total_files']} dosya")
    return True


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Bir epoch eğitim."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrikleri topla
        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch_y.cpu().tolist())
    
    # Ortalama loss
    avg_loss = total_loss / len(train_loader.dataset)
    
    # Metrikleri hesapla
    metrics = calculate_all_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validasyon."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in tqdm(val_loader, desc="Validating", leave=False):
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


def train(config: Dict[str, Any]):
    """Ana eğitim fonksiyonu."""
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    # Veri kümesi kontrolü
    data_config = config["data"]
    training_config = config["training"]
    
    if not ensure_data_exists(data_config["data_dir"]):
        raise RuntimeError("Veri kümesi hazırlanamadı!")
    
    # Veri yükle
    loaders = create_data_loaders(
        data_dir=data_config["data_dir"],
        batch_size=training_config["batch_size"],
        train_ratio=data_config["train_ratio"],
        val_ratio=data_config["val_ratio"],
        binary=data_config.get("binary_classification", False),
        random_seed=data_config.get("random_seed", 42)
    )
    
    # Model oluştur
    model = EpiGraphNet.from_config(config).to(device)
    print(f"Model oluşturuldu: EpiGraphNet")
    
    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )
    
    # Scheduler
    scheduler_config = training_config.get("scheduler", {})
    scheduler = LinearLR(
        optimizer,
        start_factor=scheduler_config.get("start_factor", 0.1),
        total_iters=scheduler_config.get("total_iters", 10)
    )
    
    # Metrik tracker
    train_tracker = MetricTracker()
    val_tracker = MetricTracker()
    
    # Checkpoint dizini
    checkpoint_dir = training_config.get("checkpoint", {}).get("save_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Early stopping
    early_stop_config = training_config.get("early_stopping", {})
    patience = early_stop_config.get("patience", 10)
    min_delta = early_stop_config.get("min_delta", 0.001)
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Eğitim döngüsü
    num_epochs = training_config["num_epochs"]
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print('='*50)
        
        # Train
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        train_tracker.update(train_metrics)
        
        # Validate
        val_metrics = validate(
            model, loaders["val"], criterion, device
        )
        val_tracker.update(val_metrics)
        
        # Scheduler step
        scheduler.step()
        
        # Print metrics
        print(f"Train: {train_tracker.summary()}")
        print(f"Val:   {val_tracker.summary()}")
        
        # Checkpoint
        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            
            # En iyi modeli kaydet
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"]
            }, checkpoint_path)
            print(f"✓ En iyi model kaydedildi: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nErken durdurma! {patience} epoch boyunca iyileşme yok.")
                break
    
    print("\n" + "="*50)
    print("Eğitim tamamlandı!")
    print(f"En iyi validasyon loss: {best_val_loss:.4f}")
    
    return model, train_tracker, val_tracker


def main():
    parser = argparse.ArgumentParser(description="EpiGraphNet Eğitimi")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Konfigürasyon dosyası yolu"
    )
    args = parser.parse_args()
    
    # Konfigürasyon yükle
    config = load_config(args.config)
    
    # Eğitim başlat
    train(config)


if __name__ == "__main__":
    main()
