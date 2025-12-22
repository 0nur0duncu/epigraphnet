import numpy as np

def normalize_signal(signal):
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True)
    
    std = np.where(std == 0, 1, std)
    
    return (signal - mean) / std


def segment_signal(
    signal: np.ndarray,
    segment_length: int,
    overlap: float = 0.0
):
    step = int(segment_length * (1 - overlap))
    n_segments = (len(signal) - segment_length) // step + 1
    
    segments = []
    for i in range(n_segments):
        start = i * step
        end = start + segment_length
        segments.append(signal[start:end])
    
    return np.array(segments)


def add_channel_dimension(signal: np.ndarray) -> np.ndarray:
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
    processed = signal.copy()
    
    if normalize:
        processed = normalize_signal(processed)
    
    if add_channel:
        processed = add_channel_dimension(processed)
    
    return processed


def load_bonn_file(filepath: str) -> np.ndarray:
    try:
        signal = np.loadtxt(filepath)
        return signal.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Dosya yüklenemedi: {filepath}. Hata: {e}")


def get_class_label(filename: str, binary: bool = False) -> int:
    prefix = filename[0].upper()

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
    return label - 1
