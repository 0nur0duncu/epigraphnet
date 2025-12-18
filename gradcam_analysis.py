"""
EpiGraphNet - Grad-CAM Analizi
CNN katmanlarının hangi EEG bölgelerine odaklandığını görselleştirir.

Kullanım:
    python gradcam_analysis.py --checkpoint checkpoints/best_model.pt
    python gradcam_analysis.py --checkpoint checkpoints/best_model.pt --num-samples 10
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Matplotlib ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# Proje modülleri
from models.epigraphnet_simple import EpiGraphNetSimple
from models import EpiGraphNet
from data.dataset import load_bonn_dataset


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementasyonu.
    1D CNN'ler için uyarlanmış versiyon.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Analiz edilecek model
            target_layer: Grad-CAM için hedef CNN katmanı
        """
        self.model = model
        self.target_layer = target_layer
        
        # Hook'lar için değişkenler
        self.gradients = None
        self.activations = None
        
        # Hook'ları kaydet
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Forward pass'te aktivasyonları kaydet."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Backward pass'te gradyanları kaydet."""
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        Grad-CAM hesapla.
        
        Args:
            x: Giriş tensörü (batch, channels, length)
            class_idx: Hedef sınıf indeksi (None ise predicted class kullanılır)
            
        Returns:
            cam: Grad-CAM haritası (batch, length)
            pred_class: Tahmin edilen sınıf
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        
        # One-hot encode target
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Grad-CAM hesapla
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=2, keepdim=True)  # (batch, channels, 1)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1)  # (batch, length)
        
        # ReLU - sadece pozitif etkileri göster
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, class_idx
    
    def remove_hooks(self):
        """Hook'ları kaldır."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def find_cnn_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Modeldeki CNN katmanlarını bul."""
    cnn_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            cnn_layers.append((name, module))
    
    return cnn_layers


def plot_gradcam_single(
    signal: np.ndarray,
    cam: np.ndarray,
    true_label: int,
    pred_label: int,
    class_names: List[str],
    sample_idx: int,
    output_dir: str
):
    """Tek bir örnek için Grad-CAM görselleştirmesi."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Orijinal sinyal
    ax1 = axes[0]
    time = np.arange(len(signal)) / 173.61  # saniye cinsinden
    ax1.plot(time, signal, 'b-', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel('Genlik (µV)')
    ax1.set_title(f'EEG Sinyali - Gerçek: {class_names[true_label]}, Tahmin: {class_names[pred_label]}')
    ax1.set_xlim([0, time[-1]])
    
    # 2. Grad-CAM haritası
    ax2 = axes[1]
    
    # CAM'i sinyal uzunluğuna upsample et
    cam_upsampled = np.interp(
        np.linspace(0, 1, len(signal)),
        np.linspace(0, 1, len(cam)),
        cam
    )
    
    ax2.imshow(cam_upsampled.reshape(1, -1), aspect='auto', cmap='jet',
               extent=[0, time[-1], 0, 1], vmin=0, vmax=1)
    ax2.set_ylabel('Grad-CAM')
    ax2.set_yticks([])
    ax2.set_title('Grad-CAM Aktivasyon Haritası')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, ax=ax2, orientation='horizontal', pad=0.1, label='Önem')
    
    # 3. Sinyal üzerine overlay
    ax3 = axes[2]
    ax3.plot(time, signal, 'b-', linewidth=0.5, alpha=0.3, label='Sinyal')
    
    # CAM'e göre renklendir
    colors = plt.cm.jet(cam_upsampled)
    for i in range(len(signal)-1):
        ax3.plot(time[i:i+2], signal[i:i+2], color=colors[i], linewidth=1.5)
    
    ax3.set_xlabel('Zaman (s)')
    ax3.set_ylabel('Genlik (µV)')
    ax3.set_title('Sinyal + Grad-CAM Overlay')
    ax3.set_xlim([0, time[-1]])
    
    plt.tight_layout()
    
    filename = f"{output_dir}/gradcam_sample_{sample_idx}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def plot_gradcam_comparison(
    signals: List[np.ndarray],
    cams: List[np.ndarray],
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    output_dir: str
):
    """Birden fazla örneği karşılaştıran Grad-CAM görselleştirmesi."""
    
    n_samples = len(signals)
    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, (signal, cam, true_label, pred_label) in enumerate(
        zip(signals, cams, true_labels, pred_labels)
    ):
        time = np.arange(len(signal)) / 173.61
        
        # CAM'i upsample et
        cam_upsampled = np.interp(
            np.linspace(0, 1, len(signal)),
            np.linspace(0, 1, len(cam)),
            cam
        )
        
        # Sol: Orijinal sinyal
        ax_left = axes[i, 0]
        ax_left.plot(time, signal, 'b-', linewidth=0.5)
        ax_left.set_ylabel('Genlik')
        ax_left.set_title(f'Sınıf: {class_names[true_label]} (Tahmin: {class_names[pred_label]})')
        
        # Sağ: Overlay
        ax_right = axes[i, 1]
        colors = plt.cm.jet(cam_upsampled)
        for j in range(len(signal)-1):
            ax_right.plot(time[j:j+2], signal[j:j+2], color=colors[j], linewidth=1)
        ax_right.set_ylabel('Genlik')
        ax_right.set_title('Grad-CAM Overlay')
    
    axes[-1, 0].set_xlabel('Zaman (s)')
    axes[-1, 1].set_xlabel('Zaman (s)')
    
    # Colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label='Önem')
    
    plt.suptitle('Grad-CAM Karşılaştırması', fontsize=14, fontweight='bold')
    
    filename = f"{output_dir}/gradcam_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def plot_class_average_gradcam(
    class_cams: Dict[int, List[np.ndarray]],
    class_names: List[str],
    output_dir: str
):
    """Her sınıf için ortalama Grad-CAM haritası."""
    
    n_classes = len(class_cams)
    fig, axes = plt.subplots(n_classes, 1, figsize=(14, 3*n_classes))
    
    if n_classes == 1:
        axes = [axes]
    
    for class_idx, ax in enumerate(axes):
        if class_idx not in class_cams or len(class_cams[class_idx]) == 0:
            ax.text(0.5, 0.5, 'Veri yok', ha='center', va='center')
            ax.set_title(f'{class_names[class_idx]}')
            continue
        
        # Tüm CAM'leri aynı uzunluğa normalize et
        target_len = max(len(cam) for cam in class_cams[class_idx])
        normalized_cams = []
        for cam in class_cams[class_idx]:
            upsampled = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(cam)),
                cam
            )
            normalized_cams.append(upsampled)
        
        # Ortalama ve std hesapla
        cams_array = np.array(normalized_cams)
        mean_cam = cams_array.mean(axis=0)
        std_cam = cams_array.std(axis=0)
        
        time = np.linspace(0, 23.6, target_len)  # ~4096 samples at 173.61 Hz
        
        ax.fill_between(time, mean_cam - std_cam, mean_cam + std_cam, alpha=0.3, color='blue')
        ax.plot(time, mean_cam, 'b-', linewidth=2, label='Ortalama')
        ax.set_ylabel('Grad-CAM')
        ax.set_title(f'{class_names[class_idx]} (n={len(class_cams[class_idx])})')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, time[-1]])
    
    axes[-1].set_xlabel('Zaman (s)')
    plt.suptitle('Sınıf Başına Ortalama Grad-CAM Aktivasyonu', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{output_dir}/gradcam_class_average.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def plot_important_regions(
    signals: List[np.ndarray],
    cams: List[np.ndarray],
    labels: List[int],
    class_names: List[str],
    output_dir: str,
    threshold: float = 0.7
):
    """Önemli bölgelerin istatistiksel analizi."""
    
    # Her sınıf için önemli bölgeleri topla
    class_important_regions = {i: [] for i in range(len(class_names))}
    
    for signal, cam, label in zip(signals, cams, labels):
        # CAM'i sinyal uzunluğuna upsample et
        cam_upsampled = np.interp(
            np.linspace(0, 1, len(signal)),
            np.linspace(0, 1, len(cam)),
            cam
        )
        
        # Önemli bölgeleri bul (threshold üstü)
        important_mask = cam_upsampled > threshold
        important_indices = np.where(important_mask)[0]
        
        if len(important_indices) > 0:
            # Bölgeleri normalize et (0-1 arası)
            normalized_positions = important_indices / len(signal)
            class_important_regions[label].extend(normalized_positions.tolist())
    
    # Histogram çiz
    fig, axes = plt.subplots(1, len(class_names), figsize=(4*len(class_names), 5))
    
    if len(class_names) == 1:
        axes = [axes]
    
    for class_idx, ax in enumerate(axes):
        positions = class_important_regions[class_idx]
        
        if len(positions) > 0:
            ax.hist(positions, bins=20, range=(0, 1), color=plt.cm.tab10(class_idx), 
                    alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Sinyal Pozisyonu (normalize)')
        ax.set_ylabel('Frekans')
        ax.set_title(f'{class_names[class_idx]}')
        ax.set_xlim([0, 1])
    
    plt.suptitle(f'Önemli Bölgelerin Dağılımı (threshold={threshold})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{output_dir}/gradcam_important_regions.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def run_gradcam_analysis(
    model: nn.Module,
    signals: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_dir: str,
    num_samples: int = 10,
    device: torch.device = None
):
    """Tam Grad-CAM analizi çalıştır."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # CNN katmanlarını bul
    cnn_layers = find_cnn_layers(model)
    
    if len(cnn_layers) == 0:
        print("❌ CNN katmanı bulunamadı!")
        return
    
    print(f"\n📋 Bulunan CNN katmanları:")
    for name, layer in cnn_layers:
        print(f"  - {name}: {layer}")
    
    # Son CNN katmanını kullan (genellikle en anlamlı)
    target_name, target_layer = cnn_layers[-1]
    print(f"\n🎯 Hedef katman: {target_name}")
    
    # Grad-CAM oluştur
    gradcam = GradCAM(model, target_layer)
    
    # Örnekler seç
    n_samples = min(num_samples, len(signals))
    indices = np.random.choice(len(signals), n_samples, replace=False)
    
    all_signals = []
    all_cams = []
    all_true_labels = []
    all_pred_labels = []
    class_cams = {i: [] for i in range(len(class_names))}
    
    print(f"\n🔬 {n_samples} örnek analiz ediliyor...")
    
    for i, idx in enumerate(indices):
        signal = signals[idx]
        label = labels[idx]
        
        # Signal'ı 1D'ye düzleştir
        if hasattr(signal, 'ndim') and signal.ndim > 1:
            signal = signal.flatten()
        signal = np.asarray(signal).flatten()
        
        # Tensor'a dönüştür: (batch=1, channels=1, length)
        x = torch.tensor(signal, dtype=torch.float32).view(1, 1, -1).to(device)
        
        # Grad-CAM hesapla
        cam, pred_class = gradcam(x)
        cam_np = cam.cpu().numpy().squeeze()
        
        all_signals.append(signal)
        all_cams.append(cam_np)
        all_true_labels.append(label)
        all_pred_labels.append(pred_class)
        class_cams[label].append(cam_np)
        
        # Tek örnek görselleştirme
        filename = plot_gradcam_single(
            signal, cam_np, label, pred_class,
            class_names, i, output_dir
        )
        print(f"  ✓ Örnek {i+1}: {filename}")
    
    # Hook'ları kaldır
    gradcam.remove_hooks()
    
    # Karşılaştırma görselleştirmesi
    print("\n📊 Karşılaştırma grafikleri oluşturuluyor...")
    
    # En fazla 5 örnek karşılaştır
    comparison_count = min(5, len(all_signals))
    filename = plot_gradcam_comparison(
        all_signals[:comparison_count],
        all_cams[:comparison_count],
        all_true_labels[:comparison_count],
        all_pred_labels[:comparison_count],
        class_names, output_dir
    )
    print(f"  ✓ Karşılaştırma: {filename}")
    
    # Sınıf ortalaması
    filename = plot_class_average_gradcam(class_cams, class_names, output_dir)
    print(f"  ✓ Sınıf ortalaması: {filename}")
    
    # Önemli bölgeler
    filename = plot_important_regions(
        all_signals, all_cams, all_true_labels,
        class_names, output_dir
    )
    print(f"  ✓ Önemli bölgeler: {filename}")
    
    print(f"\n✅ Tüm Grad-CAM analizleri '{output_dir}/' klasörüne kaydedildi.")


def main():
    parser = argparse.ArgumentParser(description="EpiGraphNet - Grad-CAM Analizi")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Model checkpoint dosyası"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Konfigürasyon dosyası"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Analiz edilecek örnek sayısı"
    )
    parser.add_argument(
        "--output-dir", type=str, default="gradcam_figures",
        help="Çıktı klasörü"
    )
    parser.add_argument(
        "--binary", action="store_true",
        help="İkili sınıflandırma için"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("EpiGraphNet - Grad-CAM Analizi")
    print("="*60)
    
    # Konfigürasyon yükle
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Veri yükle
    print("\n📂 Veri yükleniyor...")
    signals, labels = load_bonn_dataset(
        config["data"]["data_dir"],
        binary=args.binary
    )
    
    # Sınıf isimleri
    if args.binary:
        class_names = ["Normal", "Nöbet"]
    else:
        class_names = ["Set A", "Set B", "Set C", "Set D", "Set E"]
    
    print(f"  → {len(signals)} örnek yüklendi")
    print(f"  → Sınıflar: {class_names}")
    
    # Model oluştur
    num_classes = 2 if args.binary else 5
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n🔧 Checkpoint yükleniyor: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Model tipini belirle
        model = EpiGraphNetSimple(
            in_channels=1,
            conv_channels=[16, 32, 64],
            lstm_hidden=64,
            lstm_layers=2,
            num_nodes=16,
            sparsity=25.0,
            thresholding="value",
            gcn_hidden=64,
            gcn_layers=3,
            num_classes=num_classes,
            dropout=0.1
        )
        
        # State dict yükle
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("  ✓ Checkpoint yüklendi")
    else:
        print("\n⚠️ Checkpoint bulunamadı, yeni model oluşturuluyor...")
        model = EpiGraphNetSimple(
            in_channels=1,
            conv_channels=[16, 32, 64],
            lstm_hidden=64,
            lstm_layers=2,
            num_nodes=16,
            sparsity=25.0,
            thresholding="value",
            gcn_hidden=64,
            gcn_layers=3,
            num_classes=num_classes,
            dropout=0.1
        )
    
    # Grad-CAM analizi çalıştır
    run_gradcam_analysis(
        model=model,
        signals=signals,
        labels=labels,
        class_names=class_names,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=device
    )


if __name__ == "__main__":
    main()
