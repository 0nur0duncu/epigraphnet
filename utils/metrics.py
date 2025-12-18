"""
Değerlendirme Metrikleri
Doğruluk, Duyarlılık (Recall), Kesinlik (Precision), F1
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Doğruluk hesaplar.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        
    Returns:
        Doğruluk değeri (0-100)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return accuracy_score(y_true, y_pred) * 100


def calculate_precision(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    average: str = "macro"
) -> float:
    """
    Kesinlik (Precision) hesaplar.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        average: Ortalama türü ("macro", "micro", "weighted")
        
    Returns:
        Kesinlik değeri (0-100)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return precision_score(
        y_true, y_pred, average=average, zero_division=0
    ) * 100


def calculate_recall(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    average: str = "macro"
) -> float:
    """
    Duyarlılık (Recall) hesaplar.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        average: Ortalama türü
        
    Returns:
        Duyarlılık değeri (0-100)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return recall_score(
        y_true, y_pred, average=average, zero_division=0
    ) * 100


def calculate_f1(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    average: str = "macro"
) -> float:
    """
    F1 skoru hesaplar.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        average: Ortalama türü
        
    Returns:
        F1 değeri (0-100)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return f1_score(
        y_true, y_pred, average=average, zero_division=0
    ) * 100


def calculate_all_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    average: str = "macro"
) -> Dict[str, float]:
    """
    Tüm metrikleri hesaplar.
    
    Returns:
        {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
    """
    return {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "precision": calculate_precision(y_true, y_pred, average),
        "recall": calculate_recall(y_true, y_pred, average),
        "f1": calculate_f1(y_true, y_pred, average)
    }


def get_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    """Confusion matrix döndürür."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    return confusion_matrix(y_true, y_pred)


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Tensörü numpy array'e dönüştürür."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


class MetricTracker:
    """
    Eğitim boyunca metrikleri takip eder.
    """
    
    def __init__(self, metrics: List[str] = None):
        """
        Args:
            metrics: Takip edilecek metrik isimleri
        """
        self.metrics = metrics or ["loss", "accuracy", "precision", "recall", "f1"]
        self.history: Dict[str, List[float]] = {m: [] for m in self.metrics}
        self.best: Dict[str, float] = {}
    
    def update(self, values: Dict[str, float]):
        """Yeni değerler ekle."""
        for name, value in values.items():
            if name in self.history:
                self.history[name].append(value)
                
                # En iyi değeri güncelle
                if name == "loss":
                    if name not in self.best or value < self.best[name]:
                        self.best[name] = value
                else:
                    if name not in self.best or value > self.best[name]:
                        self.best[name] = value
    
    def get_last(self, name: str) -> Optional[float]:
        """Son değeri döndür."""
        if name in self.history and self.history[name]:
            return self.history[name][-1]
        return None
    
    def get_best(self, name: str) -> Optional[float]:
        """En iyi değeri döndür."""
        return self.best.get(name)
    
    def get_history(self, name: str) -> List[float]:
        """Tüm geçmişi döndür."""
        return self.history.get(name, [])
    
    def reset(self):
        """Geçmişi sıfırla."""
        self.history = {m: [] for m in self.metrics}
        self.best = {}
    
    def summary(self) -> str:
        """Özet string döndür."""
        lines = []
        for name in self.metrics:
            last = self.get_last(name)
            best = self.get_best(name)
            if last is not None:
                lines.append(f"{name}: {last:.2f} (best: {best:.2f})")
        return " | ".join(lines)
