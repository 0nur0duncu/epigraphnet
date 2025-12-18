"""
EpiGraphNet Demo
Bonn EEG veri kümesi veya sentetik veri ile modeli test eder
"""

import os
import torch
import numpy as np

from models import EpiGraphNet
from data import download_bonn_dataset, verify_dataset


def create_synthetic_eeg(
    batch_size: int = 4,
    length: int = 4097,
    num_classes: int = 5
):
    """
    Sentetik EEG verisi oluşturur.
    
    Args:
        batch_size: Örnek sayısı
        length: Sinyal uzunluğu
        num_classes: Sınıf sayısı
        
    Returns:
        (signals, labels)
    """
    # Rastgele EEG benzeri sinyal
    signals = np.random.randn(batch_size, 1, length).astype(np.float32)
    
    # Rastgele etiketler
    labels = np.random.randint(0, num_classes, size=batch_size)
    
    return torch.FloatTensor(signals), torch.LongTensor(labels)


def load_real_eeg_samples(data_dir: str = "data/bonn", num_samples: int = 4):
    """
    Gerçek Bonn EEG veri kümesinden örnek yükler.
    
    Args:
        data_dir: Veri dizini
        num_samples: Yüklenecek örnek sayısı
        
    Returns:
        (signals, labels, filenames)
    """
    from data.preprocessing import load_bonn_file, get_class_label, preprocess_eeg
    
    signals = []
    labels = []
    filenames = []
    
    prefixes = ['S', 'N', 'F', 'O', 'Z']  # Her sınıftan bir örnek
    
    for prefix in prefixes:
        if len(signals) >= num_samples:
            break
            
        # İlk dosyayı bul
        for i in range(1, 101):
            filename = f"{prefix}{i:03d}.txt"
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    signal = load_bonn_file(filepath)
                    signal = preprocess_eeg(signal, normalize=True, add_channel=True)
                    label = get_class_label(filename, binary=False)
                    
                    signals.append(signal)
                    labels.append(label)
                    filenames.append(filename)
                    break
                except Exception as e:
                    print(f"  Uyarı: {filename} yüklenemedi: {e}")
    
    if len(signals) == 0:
        return None, None, None
    
    signals = np.array(signals)
    labels = np.array(labels)
    
    return (
        torch.FloatTensor(signals),
        torch.LongTensor(labels),
        filenames
    )


def main():
    print("="*60)
    print("EpiGraphNet Demo")
    print("="*60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz: {device}")
    
    # Veri kümesi kontrolü
    data_dir = "data/bonn"
    use_real_data = False
    
    stats = verify_dataset(data_dir)
    if 'error' not in stats and stats.get('is_valid', False):
        print(f"\n✓ Bonn veri kümesi mevcut: {stats['total_files']} dosya")
        use_real_data = True
    else:
        print(f"\n⚠ Bonn veri kümesi bulunamadı. Sentetik veri kullanılacak.")
        print("  Gerçek veri için: python data/download_bonn.py")
    
    # Model oluştur (Şekil 1'e göre)
    print("\n1. Model oluşturuluyor (Şekil 1'e uygun)...")
    model = EpiGraphNet(
        # CNN (Şekil 1: Conv1D -> MaxPool -> Conv1D -> Conv1D)
        in_channels=1,
        conv_channels=[16, 32, 64],
        kernel_sizes=[5, 5, 5],
        pool_size=2,  # Sadece ilk katmanda MaxPool
        fc_hidden=128,
        # LSTM (Şekil 1: 2 LSTM katmanı)
        lstm_hidden=64,
        lstm_layers=2,
        sequence_length=8,  # T
        # Graph
        num_windows=8,
        num_nodes=16,
        sparsity=50.0,
        thresholding="value",  # Değer eşikleme (DE)
        # GCN (Şekil 1: 3 GraphConv katmanı)
        gcn_hidden=64,
        gcn_layers=3,
        # Classification
        num_classes=5,
        dropout=0.1
    ).to(device)
    
    # Model parametrelerini say
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Toplam parametre: {total_params:,}")
    print(f"   Eğitilebilir: {trainable_params:,}")
    
    # Veri yükle
    if use_real_data:
        print("\n2. Gerçek Bonn EEG verisi yükleniyor...")
        signals, labels, filenames = load_real_eeg_samples(data_dir, num_samples=5)
        
        if signals is not None:
            batch_size = len(signals)
            print(f"   Yüklenen dosyalar: {filenames}")
            
            # Sınıf isimleri
            class_names = {
                0: "Nöbet (S)",
                1: "Epilepsi-Nötr (N)",
                2: "Epilepsi-Uyaran (F)",
                3: "Sağlıklı-Kapalı (O)",
                4: "Sağlıklı-Açık (Z)"
            }
            label_names = [class_names.get(l.item(), "?") for l in labels]
            print(f"   Sınıflar: {label_names}")
        else:
            print("   Gerçek veri yüklenemedi, sentetik veriye geçiliyor...")
            use_real_data = False
    
    if not use_real_data:
        print("\n2. Sentetik EEG verisi oluşturuluyor...")
        batch_size = 4
        signals, labels = create_synthetic_eeg(
            batch_size=batch_size,
            length=4097,  # Bonn veri kümesi uzunluğu
            num_classes=5
        )
    
    signals = signals.to(device)
    labels = labels.to(device)
    print(f"   Sinyal boyutu: {signals.shape}")
    print(f"   Etiketler: {labels.tolist()}")
    
    # Forward pass
    print("\n3. Forward pass yapılıyor...")
    model.eval()
    with torch.no_grad():
        logits = model(signals)
    
    print(f"   Çıkış boyutu: {logits.shape}")
    
    # Tahminler
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    
    print(f"   Tahminler: {preds.tolist()}")
    print(f"   Gerçek: {labels.tolist()}")
    
    # Olasılıklar
    print("\n4. Sınıf olasılıkları:")
    class_labels = ["Nöbet", "Epi-N", "Epi-F", "Sağ-K", "Sağ-A"]
    print(f"   Sınıflar: {class_labels}")
    for i in range(len(signals)):
        prob_str = ", ".join([f"{p:.2f}" for p in probs[i].tolist()])
        print(f"   Örnek {i+1}: [{prob_str}] -> Sınıf {preds[i].item()}")
    
    # Model bileşenlerini göster
    print("\n5. Model bileşenleri:")
    print("   ├── CNN-LSTM Modülü")
    print("   │   ├── CNN Encoder (3 blok)")
    print("   │   └── LSTM (2 katman)")
    print("   ├── Graph Builder")
    print("   │   ├── KBM Hesaplayıcı")
    print("   │   └── Değer Eşikleme (DE)")
    print("   └── GCN Classifier")
    print("       ├── GraphConv (3 katman)")
    print("       ├── Global Max Pool")
    print("       └── FC + Softmax")
    
    print("\n" + "="*60)
    print("✓ Demo başarıyla tamamlandı!")
    print("="*60)
    
    # Bonus: Farklı eşikleme yöntemini de test et
    print("\n[Bonus] Bağlantı Eşikleme (BE) ile test:")
    model_be = EpiGraphNet(
        thresholding="connection",
        sparsity=25.0,
        num_classes=5
    ).to(device)
    
    with torch.no_grad():
        logits_be = model_be(signals)
        preds_be = logits_be.argmax(dim=1)
    
    print(f"   BE Tahminleri: {preds_be.tolist()}")
    print("   ✓ BE modeli de çalışıyor!")


if __name__ == "__main__":
    main()
