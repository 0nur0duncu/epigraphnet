"""
Görselleştirme Fonksiyonları
Eğitim eğrileri, confusion matrix, korelasyon matrisi
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional


def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: List[str] = ["loss", "accuracy"],
    title: str = "Eğitim Eğrileri",
    save_path: Optional[str] = None
):
    """
    Eğitim eğrilerini çizer.
    
    Args:
        history: Metrik geçmişi {"loss": [...], "accuracy": [...]}
        metrics: Çizilecek metrikler
        title: Grafik başlığı
        save_path: Kayıt yolu (opsiyonel)
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], marker='o', markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{metric.capitalize()} vs Epoch")
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True
):
    """
    Confusion matrix çizer.
    
    Args:
        cm: Confusion matrix (n_classes, n_classes)
        class_names: Sınıf isimleri
        title: Grafik başlığı
        save_path: Kayıt yolu
        normalize: Normalize et
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"Sınıf {i+1}" for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gerçek')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(
    C: np.ndarray,
    title: str = "Korelasyon Matrisi (KBM)",
    save_path: Optional[str] = None
):
    """
    Korelasyon matrisini görselleştirir.
    
    Args:
        C: Korelasyon matrisi (N, N)
        title: Grafik başlığı
        save_path: Kayıt yolu
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        C,
        annot=False,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )
    
    ax.set_xlabel('Düğüm j')
    ax.set_ylabel('Düğüm i')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_graph_adjacency(
    adjacency: np.ndarray,
    title: str = "Grafik Adjacency Matrix",
    save_path: Optional[str] = None
):
    """
    Grafik adjacency matrisini görselleştirir.
    
    Args:
        adjacency: Adjacency matrix (N, N)
        title: Grafik başlığı
        save_path: Kayıt yolu
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    sns.heatmap(
        adjacency,
        annot=False,
        cmap='Greys',
        cbar=True,
        square=True,
        ax=ax
    )
    
    ax.set_xlabel('Düğüm j')
    ax.set_ylabel('Düğüm i')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_eeg_signal(
    signal: np.ndarray,
    sampling_rate: float = 173.61,
    title: str = "EEG Sinyali",
    save_path: Optional[str] = None
):
    """
    EEG sinyalini çizer.
    
    Args:
        signal: EEG sinyali (length,)
        sampling_rate: Örnekleme frekansı (Hz)
        title: Grafik başlığı
        save_path: Kayıt yolu
    """
    length = len(signal)
    time = np.arange(length) / sampling_rate
    
    fig, ax = plt.subplots(figsize=(12, 3))
    
    ax.plot(time, signal, linewidth=0.5)
    ax.set_xlabel('Zaman (s)')
    ax.set_ylabel('Amplitüd')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
