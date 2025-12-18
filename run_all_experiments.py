"""
EpiGraphNet - Tüm Deneyler
Makaledeki Tablo II'deki tüm sonuçları üretir.

Deneyler:
1. İki Sınıflı (Binary): Nöbet var (1) vs Nöbet yok (0)
2. Çok Sınıflı (Multi-class): 5 sınıf

Modeller:
- CNN-LSTM (baseline)
- 1D-CNN-LSTM (baseline)  
- EpiGraphNet_DE (a=50)
- EpiGraphNet_DE (a=25)
- EpiGraphNet_BE (a=50)
- EpiGraphNet_BE (a=25)

Her model 5 kez çalıştırılır ve ortalama alınır.
"""

import os
import sys
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
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime
from tabulate import tabulate

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100

from models import EpiGraphNet
from models.cnn_lstm import CNNLSTM
from data import BonnEEGDataset
from data.dataset import load_bonn_dataset
from utils import calculate_all_metrics


# ============================================================================
# Baseline Modeller
# ============================================================================

class BaselineCNNLSTM(nn.Module):
    """
    Baseline CNN-LSTM modeli (EpiGraphNet'in CNN-LSTM kısmı, GCN olmadan).
    Referans: [16] Xu et al., 2020
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [5, 5, 5],
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # CNN katmanları
        cnn_layers = []
        in_ch = in_channels
        for i, (out_ch, ks) in enumerate(zip(conv_channels, kernel_sizes)):
            cnn_layers.append(nn.Conv1d(in_ch, out_ch, ks, padding=ks//2))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(2))
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, length)
        
        # CNN
        x = self.cnn(x)  # (batch, channels, length')
        
        # Adaptive pool
        x = self.adaptive_pool(x)  # (batch, channels, 64)
        
        # LSTM için yeniden şekillendir
        x = x.permute(0, 2, 1)  # (batch, 64, channels)
        
        # LSTM
        x, _ = self.lstm(x)  # (batch, 64, hidden)
        
        # Son zaman adımı
        x = x[:, -1, :]  # (batch, hidden)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class Baseline1DCNNLSTM(nn.Module):
    """
    1D-CNN-LSTM modeli.
    Referans: [17] Shanmugam & Dharmar, 2023
    Daha derin CNN + Bidirectional LSTM
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64, 128, 64],
        kernel_sizes: List[int] = [7, 5, 5, 3],
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Daha derin CNN
        cnn_layers = []
        in_ch = in_channels
        for i, (out_ch, ks) in enumerate(zip(conv_channels, kernel_sizes)):
            cnn_layers.append(nn.Conv1d(in_ch, out_ch, ks, padding=ks//2))
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(2))
            cnn_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        x = self.cnn(x)
        
        # Adaptive pool
        x = self.adaptive_pool(x)
        
        # LSTM için yeniden şekillendir
        x = x.permute(0, 2, 1)
        
        # Bidirectional LSTM
        x, _ = self.lstm(x)
        
        # Son zaman adımı
        x = x[:, -1, :]
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# ============================================================================
# Eğitim ve Değerlendirme Fonksiyonları
# ============================================================================

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


def create_model(
    model_type: str,
    num_classes: int,
    sparsity: float = 50,
    thresholding: str = "value",
    config: Dict[str, Any] = None
) -> nn.Module:
    """Model oluşturur."""
    
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
        # Basitleştirilmiş model kullan - daha stabil
        from models.epigraphnet_simple import EpiGraphNetSimple
        return EpiGraphNetSimple(
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
    
    else:
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
    """Tek bir deney çalıştırır."""
    
    # Seed ayarla
    seed = 42 + run_id * 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Sınıf sayısı
    num_classes = 2 if binary else 5
    
    # Veri bölme
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        signals, labels,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=labels
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=seed,
        stratify=y_temp
    )
    
    # DataLoader'lar
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        BonnEEGDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        BonnEEGDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        BonnEEGDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Model oluştur
    model = create_model(
        model_type, num_classes, sparsity, thresholding, config
    ).to(device)
    
    # Eğitim ayarları
    training_config = config["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )
    
    # Scheduler - Cosine Annealing daha stabil
    from torch.optim.lr_scheduler import CosineAnnealingLR
    num_epochs = training_config["num_epochs"]
    
    # EpiGraphNet için daha fazla epoch
    if model_type.startswith("EpiGraphNet"):
        num_epochs = max(num_epochs, 100)  # En az 100 epoch
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Early stopping - EpiGraphNet için daha sabırlı
    patience = training_config.get("early_stopping", {}).get("patience", 15)
    if model_type.startswith("EpiGraphNet"):
        patience = 25  # Daha sabırlı
    
    min_delta = training_config.get("early_stopping", {}).get("min_delta", 0.001)
    best_val_acc = 0.0  # Loss yerine accuracy kullan
    patience_counter = 0
    best_state = None
    
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Accuracy'ye göre best model seç
        if val_metrics["accuracy"] > best_val_acc + min_delta:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # En iyi modeli yükle
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Test
    test_metrics = validate(model, test_loader, criterion, device)
    
    return test_metrics


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
    """Bir deney setini çalıştırır (5 run)."""
    
    all_results = []
    
    classification_type = "İkili" if binary else "Çok Sınıflı"
    model_name = model_type
    if "EpiGraphNet" in model_type:
        th_str = "DE" if thresholding == "value" else "BE"
        model_name = f"EpiGraphNet_{th_str}(a={int(sparsity)})"
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Sınıflandırma: {classification_type}")
    print(f"{'='*60}")
    
    for run_id in range(1, num_runs + 1):
        print(f"  Run {run_id}/{num_runs}...", end=" ", flush=True)
        
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
        print(f"Acc: {metrics['accuracy']:.2f}%")
    
    # Ortalama hesapla
    avg_results = {}
    for key in all_results[0].keys():
        if key != "loss":
            values = [r[key] for r in all_results]
            avg_results[key] = np.mean(values)
            avg_results[f"{key}_std"] = np.std(values)
    
    print(f"\n  Ortalama: Acc={avg_results['accuracy']:.2f}%, "
          f"Recall={avg_results['recall']:.2f}%, "
          f"Precision={avg_results['precision']:.2f}%, "
          f"F1={avg_results['f1']:.2f}%")
    
    return avg_results


# ============================================================================
# Görselleştirme Fonksiyonları
# ============================================================================

def plot_accuracy_comparison(all_results: Dict, paper_results: Dict, output_dir: str = "figures"):
    """
    Model doğruluklarını karşılaştıran bar chart.
    Makale sonuçları ile yan yana gösterir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in all_results["experiments"]:
        class_label = "İkili Sınıflandırma" if class_name == "binary" else "Çok Sınıflı Sınıflandırma"
        
        models = list(all_results["experiments"][class_name].keys())
        our_acc = [all_results["experiments"][class_name][m]["accuracy"] for m in models]
        paper_acc = [paper_results.get(class_name, {}).get(m, {}).get("accuracy", 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        bars1 = ax.bar(x - width/2, paper_acc, width, label='Makale', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, our_acc, width, label='Bizim', color='#4CAF50', alpha=0.8)
        
        ax.set_ylabel('Doğruluk (%)')
        ax.set_title(f'{class_label} - Model Karşılaştırması')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 105])
        
        # Değerleri bar üstüne yaz
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = f"{output_dir}/accuracy_comparison_{class_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Kaydedildi: {filename}")


def plot_metrics_heatmap(all_results: Dict, output_dir: str = "figures"):
    """
    Tüm metrikleri gösteren heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in all_results["experiments"]:
        class_label = "İkili Sınıflandırma" if class_name == "binary" else "Çok Sınıflı Sınıflandırma"
        
        models = list(all_results["experiments"][class_name].keys())
        metrics = ["accuracy", "recall", "precision", "f1"]
        metric_labels = ["Doğruluk", "Duyarlılık", "Kesinlik", "F1"]
        
        data = []
        for model in models:
            row = [all_results["experiments"][class_name][model].get(m, 0) for m in metrics]
            data.append(row)
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            data, 
            annot=True, 
            fmt='.2f',
            cmap='RdYlGn',
            xticklabels=metric_labels,
            yticklabels=models,
            vmin=0,
            vmax=100,
            ax=ax,
            linewidths=0.5,
            cbar_kws={'label': 'Değer (%)'}
        )
        
        ax.set_title(f'{class_label} - Metrik Değerleri')
        plt.tight_layout()
        filename = f"{output_dir}/metrics_heatmap_{class_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Kaydedildi: {filename}")


def plot_model_radar(all_results: Dict, output_dir: str = "figures"):
    """
    Her model için radar/spider chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ["accuracy", "recall", "precision", "f1"]
    metric_labels = ["Doğruluk", "Duyarlılık", "Kesinlik", "F1"]
    
    for class_name in all_results["experiments"]:
        class_label = "İkili Sınıflandırma" if class_name == "binary" else "Çok Sınıflı Sınıflandırma"
        
        models = list(all_results["experiments"][class_name].keys())
        n_models = len(models)
        
        # Renk paleti
        colors = sns.color_palette("husl", n_models)
        
        # Radar chart için açılar
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Kapatmak için
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, model in enumerate(models):
            values = [all_results["experiments"][class_name][model].get(m, 0) for m in metrics]
            values += values[:1]  # Kapatmak için
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 100)
        ax.set_title(f'{class_label} - Model Performansları', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        filename = f"{output_dir}/radar_chart_{class_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Kaydedildi: {filename}")


def plot_difference_chart(all_results: Dict, paper_results: Dict, output_dir: str = "figures"):
    """
    Makale ile aramızdaki farkı gösteren chart.
    Yeşil = biz daha iyi, Kırmızı = makale daha iyi
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in all_results["experiments"]:
        class_label = "İkili Sınıflandırma" if class_name == "binary" else "Çok Sınıflı Sınıflandırma"
        
        models = list(all_results["experiments"][class_name].keys())
        differences = []
        
        for model in models:
            our_acc = all_results["experiments"][class_name][model]["accuracy"]
            paper_acc = paper_results.get(class_name, {}).get(model, {}).get("accuracy", our_acc)
            differences.append(our_acc - paper_acc)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#4CAF50' if d >= 0 else '#F44336' for d in differences]
        bars = ax.barh(models, differences, color=colors, alpha=0.8)
        
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Fark (Bizim - Makale) %')
        ax.set_title(f'{class_label} - Makale ile Karşılaştırma')
        
        # Değerleri bar yanına yaz
        for bar, diff in zip(bars, differences):
            width = bar.get_width()
            ax.annotate(f'{diff:+.2f}%',
                        xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(5 if width >= 0 else -5, 0),
                        textcoords="offset points",
                        ha='left' if width >= 0 else 'right',
                        va='center', fontsize=10)
        
        plt.tight_layout()
        filename = f"{output_dir}/difference_chart_{class_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Kaydedildi: {filename}")


def plot_binary_vs_multiclass(all_results: Dict, output_dir: str = "figures"):
    """
    Binary ve Multi-class sonuçlarını karşılaştıran grouped bar chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if "binary" not in all_results["experiments"] or "multiclass" not in all_results["experiments"]:
        print("  ⚠ Binary veya multiclass sonuçları eksik, bu grafik atlanıyor.")
        return
    
    models = list(all_results["experiments"]["binary"].keys())
    binary_acc = [all_results["experiments"]["binary"][m]["accuracy"] for m in models]
    multi_acc = [all_results["experiments"]["multiclass"][m]["accuracy"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width/2, binary_acc, width, label='İkili Sınıflandırma', color='#3F51B5', alpha=0.8)
    bars2 = ax.bar(x + width/2, multi_acc, width, label='Çok Sınıflı', color='#FF9800', alpha=0.8)
    
    ax.set_ylabel('Doğruluk (%)')
    ax.set_title('İkili vs Çok Sınıflı Sınıflandırma Karşılaştırması')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 105])
    
    # Değerleri bar üstüne yaz
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = f"{output_dir}/binary_vs_multiclass.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Kaydedildi: {filename}")


def plot_all_results(all_results: Dict, paper_results: Dict, output_dir: str = "figures"):
    """
    Tüm görselleştirmeleri oluşturur.
    """
    print("\n" + "="*60)
    print("GÖRSELLEŞTİRMELER")
    print("="*60)
    
    print("\n📊 Doğruluk karşılaştırması oluşturuluyor...")
    plot_accuracy_comparison(all_results, paper_results, output_dir)
    
    print("\n🔥 Metrik heatmap oluşturuluyor...")
    plot_metrics_heatmap(all_results, output_dir)
    
    print("\n🎯 Radar chart oluşturuluyor...")
    plot_model_radar(all_results, output_dir)
    
    print("\n📉 Fark grafiği oluşturuluyor...")
    plot_difference_chart(all_results, paper_results, output_dir)
    
    print("\n📈 Binary vs Multiclass karşılaştırması oluşturuluyor...")
    plot_binary_vs_multiclass(all_results, output_dir)
    
    print(f"\n✅ Tüm grafikler '{output_dir}/' klasörüne kaydedildi.")


def main():
    parser = argparse.ArgumentParser(description="EpiGraphNet - Tüm Deneyler")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Konfigürasyon dosyası"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Veri dizini"
    )
    parser.add_argument(
        "--num-runs", type=int, default=5,
        help="Her deney için çalıştırma sayısı"
    )
    parser.add_argument(
        "--output", type=str, default="all_experiments_results.json",
        help="Sonuç dosyası"
    )
    parser.add_argument(
        "--binary-only", action="store_true",
        help="Sadece ikili sınıflandırma"
    )
    parser.add_argument(
        "--multiclass-only", action="store_true",
        help="Sadece çok sınıflı sınıflandırma"
    )
    args = parser.parse_args()
    
    # Konfigürasyon yükle
    config = load_config(args.config)
    
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Deney konfigürasyonları
    experiments = [
        # (model_type, sparsity, thresholding)
        ("CNN-LSTM", None, None),
        ("1D-CNN-LSTM", None, None),
        ("EpiGraphNet", 50, "value"),    # DE (a=50)
        ("EpiGraphNet", 25, "value"),    # DE (a=25)
        ("EpiGraphNet", 50, "connection"),  # BE (a=50)
        ("EpiGraphNet", 25, "connection"),  # BE (a=25)
    ]
    
    # Sınıflandırma türleri
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
    
    # Her sınıflandırma türü için
    for class_name, binary in classification_types:
        print(f"\n{'#'*70}")
        print(f"# {'İKİLİ SINIFLANDIRMA' if binary else 'ÇOK SINIFLI SINIFLANDIRMA'}")
        print(f"{'#'*70}")
        
        # Veri yükle
        signals, labels = load_bonn_dataset(
            config["data"]["data_dir"],
            binary=binary
        )
        print(f"\nVeri: {len(signals)} örnek, {len(np.unique(labels))} sınıf")
        
        all_results["experiments"][class_name] = {}
        
        # Her model için
        for model_type, sparsity, thresholding in experiments:
            # Model adı
            if model_type == "EpiGraphNet":
                th_str = "DE" if thresholding == "value" else "BE"
                model_name = f"EpiGraphNet_{th_str}(a={sparsity})"
            else:
                model_name = model_type
            
            # Deneyi çalıştır
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
    
    # Sonuçları tabloya dönüştür
    print("\n" + "="*80)
    print("SONUÇ TABLOSU (Tablo II)")
    print("="*80)
    
    for class_name in all_results["experiments"]:
        class_label = "İki Sınıflı" if class_name == "binary" else "Çok Sınıflı"
        print(f"\n{class_label} Sınıflandırma:")
        print("-"*70)
        
        table_data = []
        for model_name, results in all_results["experiments"][class_name].items():
            table_data.append([
                model_name,
                f"{results['accuracy']:.2f}",
                f"{results['recall']:.2f}",
                f"{results['precision']:.2f}",
                f"{results['f1']:.2f}"
            ])
        
        headers = ["Model", "Doğruluk", "Duyarlılık", "Kesinlik", "F1"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Makale karşılaştırması
    print("\n" + "="*80)
    print("MAKALE KARŞILAŞTIRMASI (Tablo II)")
    print("="*80)
    
    paper_results = {
        "binary": {
            "CNN-LSTM": {"accuracy": 99.30, "recall": 98.69, "precision": 98.70, "f1": 98.62},
            "1D-CNN-LSTM": {"accuracy": 99.04, "recall": 97.69, "precision": 99.07, "f1": 98.30},
            "EpiGraphNet_DE(a=50)": {"accuracy": 99.30, "recall": 98.71, "precision": 99.12, "f1": 98.89},
            "EpiGraphNet_DE(a=25)": {"accuracy": 99.56, "recall": 99.31, "precision": 99.45, "f1": 99.36},
            "EpiGraphNet_BE(a=50)": {"accuracy": 99.00, "recall": 97.68, "precision": 98.96, "f1": 98.22},
            "EpiGraphNet_BE(a=25)": {"accuracy": 99.12, "recall": 98.15, "precision": 99.15, "f1": 98.63},
        },
        "multiclass": {
            "CNN-LSTM": {"accuracy": 77.70, "recall": 77.49, "precision": 78.70, "f1": 76.97},
            "1D-CNN-LSTM": {"accuracy": 79.50, "recall": 79.55, "precision": 79.96, "f1": 78.91},
            "EpiGraphNet_DE(a=50)": {"accuracy": 80.60, "recall": 80.28, "precision": 81.26, "f1": 79.60},
            "EpiGraphNet_DE(a=25)": {"accuracy": 82.47, "recall": 81.96, "precision": 82.13, "f1": 81.33},
            "EpiGraphNet_BE(a=50)": {"accuracy": 81.26, "recall": 80.87, "precision": 81.30, "f1": 80.34},
            "EpiGraphNet_BE(a=25)": {"accuracy": 82.51, "recall": 81.98, "precision": 82.25, "f1": 82.14},
        }
    }
    
    for class_name in all_results["experiments"]:
        class_label = "İki Sınıflı" if class_name == "binary" else "Çok Sınıflı"
        print(f"\n{class_label} Sınıflandırma - Karşılaştırma:")
        print("-"*70)
        
        table_data = []
        for model_name, our_results in all_results["experiments"][class_name].items():
            paper = paper_results.get(class_name, {}).get(model_name, {})
            if paper:
                diff = our_results['accuracy'] - paper['accuracy']
                status = "✓" if diff >= -2 else "△"
                table_data.append([
                    model_name,
                    f"{paper['accuracy']:.2f}",
                    f"{our_results['accuracy']:.2f}",
                    f"{diff:+.2f}",
                    status
                ])
        
        headers = ["Model", "Makale", "Bizim", "Fark", ""]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Görselleştirmeler oluştur
    plot_all_results(all_results, paper_results, output_dir="figures")
    
    # Sonuçları kaydet
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Sonuçlar kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
