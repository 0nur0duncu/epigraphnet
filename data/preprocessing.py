"""
EEG Sinyal Önişleme Modülü
Makaledeki Bölüm II.A - Veri Kümesi Önişlemesi
"""

import numpy as np
from typing import Tuple, Optional


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    EEG sinyalini z-score normalizasyonu ile normalize eder.
    
    Args:
        signal: Ham EEG sinyali (d,) veya (n_samples, d)
        
    Returns:
        Normalize edilmiş sinyal
    """
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True)
    
    # Sıfıra bölme kontrolü
    std = np.where(std == 0, 1, std)
    
    return (signal - mean) / std


def segment_signal(
    signal: np.ndarray,
    segment_length: int,
    overlap: float = 0.0
) -> np.ndarray:
    """
    Uzun EEG sinyalini segmentlere böler.
    
    Args:
        signal: EEG sinyali (d,)
        segment_length: Her segmentin uzunluğu
        overlap: Segmentler arası örtüşme oranı (0-1)
        
    Returns:
        Segmentlenmiş sinyal (n_segments, segment_length)
    """
    step = int(segment_length * (1 - overlap))
    n_segments = (len(signal) - segment_length) // step + 1
    
    segments = []
    for i in range(n_segments):
        start = i * step
        end = start + segment_length
        segments.append(signal[start:end])
    
    return np.array(segments)


def add_channel_dimension(signal: np.ndarray) -> np.ndarray:
    """
    1D CNN için kanal boyutu ekler.
    Makaledeki Eşitlik 1: x_i ∈ R^d -> x̄_i ∈ R^(1×d)
    
    Args:
        signal: EEG sinyali (d,) veya (batch, d)
        
    Returns:
        Kanal boyutu eklenmiş sinyal (1, d) veya (batch, 1, d)
    """
    if signal.ndim == 1:
        return signal.reshape(1, -1)
    elif signal.ndim == 2:
        return signal[:, np.newaxis, :]
    return signal


def preprocess_eeg(
    signal: np.ndarray,
    normalize: bool = True,
    add_channel: bool = True
) -> np.ndarray:
    """
    Tam EEG önişleme pipeline'ı.
    
    Args:
        signal: Ham EEG sinyali
        normalize: Normalizasyon uygula
        add_channel: Kanal boyutu ekle
        
    Returns:
        İşlenmiş EEG sinyali
    """
    processed = signal.copy()
    
    if normalize:
        processed = normalize_signal(processed)
    
    if add_channel:
        processed = add_channel_dimension(processed)
    
    return processed


def load_bonn_file(filepath: str) -> np.ndarray:
    """
    Bonn veri kümesinden tek bir dosya yükler.
    
    Not: Resmi Bonn veri kümesi (UPF) her dosyada 4096 örnek içerir.
    (23.6 saniye × 173.61 Hz ≈ 4096)
    Bazı GitHub kaynakları 4097 örnek içerebilir.
    
    Args:
        filepath: Dosya yolu (.txt formatı)
        
    Returns:
        EEG sinyali (4096 veya 4097,)
    """
    try:
        signal = np.loadtxt(filepath)
        return signal.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Dosya yüklenemedi: {filepath}. Hata: {e}")


def get_class_label(filename: str, binary: bool = False) -> int:
    """
    Dosya adından sınıf etiketini çıkarır.
    
    Bonn veri kümesi yapısı:
    - Set A (Z): Sağlıklı, gözler açık -> Sınıf 5 (veya 0 ikili)
    - Set B (O): Sağlıklı, gözler kapalı -> Sınıf 4 (veya 0 ikili)
    - Set C (N): Epilepsi, nöbet yok, nötr -> Sınıf 2 (veya 0 ikili)
    - Set D (F): Epilepsi, nöbet yok, uyaran -> Sınıf 3 (veya 0 ikili)
    - Set E (S): Epilepsi, nöbet var -> Sınıf 1 (veya 1 ikili)
    
    Args:
        filename: Dosya adı (örn: "Z001.txt")
        binary: İkili sınıflandırma mı
        
    Returns:
        Sınıf etiketi
    """
    prefix = filename[0].upper()
    
    # Çok sınıflı etiketler (1-5)
    class_map = {
        'S': 1,  # Nöbet var
        'N': 2,  # Epilepsi - nötr
        'F': 3,  # Epilepsi - uyaran
        'O': 4,  # Sağlıklı - gözler kapalı
        'Z': 5,  # Sağlıklı - gözler açık
    }
    
    label = class_map.get(prefix, 0)
    
    if binary:
        # Nöbet var (1) vs Nöbet yok (0)
        return 1 if label == 1 else 0
    
    # 0-indexed için 1 çıkar
    return label - 1
