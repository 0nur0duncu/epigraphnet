"""
Grafik Oluşturma Modülü
Makaledeki Bölüm II.C - KBM ile Grafik Oluşturulması
Eşitlik 6-13: Korelasyon hesaplama ve eşikleme yöntemleri
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal


class CorrelationMatrixBuilder(nn.Module):
    """
    Korelasyonel Bağlantı Matrisi (KBM) hesaplayıcı.
    
    LSTM çıktısını pencereler ve düğümlere bölerek
    her pencere için korelasyon matrisi hesaplar.
    
    Eşitlik 6-10'u uygular.
    """
    
    def __init__(
        self,
        num_windows: int = 8,
        num_nodes: int = 16,
        eps: float = 1e-8
    ):
        """
        Args:
            num_windows: Pencere sayısı (W)
            num_nodes: Düğüm sayısı (N)
            eps: Sayısal kararlılık için epsilon
        """
        super().__init__()
        
        self.num_windows = num_windows
        self.num_nodes = num_nodes
        self.eps = eps
    
    def forward(
        self,
        H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LSTM çıktısından KBM hesaplar.
        
        Args:
            H: LSTM çıktısı (batch, T, hidden_size)
            
        Returns:
            X: Düğüm öznitelikleri (batch, N, W * T_w)
            C: Korelasyon matrisleri (batch, W, N, N)
        """
        batch_size, T, hidden_size = H.shape
        
        # Toplam öznitelik boyutunu hesapla
        total_features = T * hidden_size
        
        # H'yi düzleştir: (batch, T * hidden_size)
        H_flat = H.reshape(batch_size, -1)
        
        # W pencereye böl
        # Her pencere T_w = total_features / W öznitelik içerir
        features_per_window = total_features // self.num_windows
        
        # Boyut uyumluluğu için padding gerekebilir
        if total_features % self.num_windows != 0:
            pad_size = self.num_windows - (total_features % self.num_windows)
            H_flat = torch.nn.functional.pad(H_flat, (0, pad_size))
            features_per_window = H_flat.shape[1] // self.num_windows
        
        # Pencerelere böl: (batch, W, features_per_window)
        H_windowed = H_flat.reshape(batch_size, self.num_windows, -1)
        
        # Her pencereyi N düğüme böl
        # Her düğüm T_w = features_per_window / N öznitelik içerir
        features_per_node = features_per_window // self.num_nodes
        
        if features_per_window % self.num_nodes != 0:
            pad_size = self.num_nodes - (features_per_window % self.num_nodes)
            H_windowed = torch.nn.functional.pad(H_windowed, (0, pad_size))
            features_per_node = H_windowed.shape[2] // self.num_nodes
        
        # X: (batch, W, N, T_w)
        X = H_windowed.reshape(batch_size, self.num_windows, self.num_nodes, -1)
        
        # KBM hesapla
        C = self._compute_correlation_matrices(X)
        
        # Düğüm özniteliklerini yeniden şekillendir: (batch, N, W * T_w)
        X_nodes = X.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        
        return X_nodes, C
    
    def _compute_correlation_matrices(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Korelasyon matrislerini hesaplar (Eşitlik 6-10).
        
        Args:
            X: Pencerelenmiş öznitelikler (batch, W, N, T_w)
            
        Returns:
            C: Korelasyon matrisleri (batch, W, N, N)
        """
        batch_size, W, N, T_w = X.shape
        
        # Eşitlik 6: Zaman boyutunda ortalama
        # x̄_i^(k) = (1/T) * Σ x_i^(k)(t)
        mean = X.mean(dim=-1, keepdim=True)  # (batch, W, N, 1)
        
        # Eşitlik 7: Merkezileştirme
        # x̃_i^(k)(t) = x_i^(k)(t) - x̄_i^(k)
        X_centered = X - mean  # (batch, W, N, T_w)
        
        # Eşitlik 8: Kovaryans matrisi
        # V_ij^(k) = (1/(T-1)) * Σ x̃_i^(k)(t) * x̃_j^(k)(t)
        # Matris çarpımı ile: V = X̃ @ X̃^T / (T-1)
        covariance = torch.matmul(X_centered, X_centered.transpose(-2, -1))
        covariance = covariance / (T_w - 1 + self.eps)  # (batch, W, N, N)
        
        # Eşitlik 9: Standart sapma
        # σ_i^(k) = sqrt(V_ii^(k))
        variance = torch.diagonal(covariance, dim1=-2, dim2=-1)  # (batch, W, N)
        std = torch.sqrt(variance + self.eps)  # (batch, W, N)
        
        # Eşitlik 10: Korelasyon
        # C_ij^(k) = V_ij^(k) / (σ_i^(k) * σ_j^(k))
        std_outer = std.unsqueeze(-1) * std.unsqueeze(-2)  # (batch, W, N, N)
        C = covariance / (std_outer + self.eps)
        
        # Korelasyon değerlerini [-1, 1] aralığına sınırla
        C = torch.clamp(C, -1.0, 1.0)
        
        return C


class GraphBuilder(nn.Module):
    """
    Grafik Oluşturucu.
    
    KBM'leri eşikleyerek kenar bağlantılarını belirler.
    Değer Eşikleme (DE) veya Bağlantı Eşikleme (BE) destekler.
    """
    
    def __init__(
        self,
        num_windows: int = 8,
        num_nodes: int = 16,
        sparsity: float = 50.0,
        thresholding: Literal["value", "connection"] = "value"
    ):
        """
        Args:
            num_windows: Pencere sayısı (W)
            num_nodes: Düğüm sayısı (N)
            sparsity: Seyreklik parametresi (a) [0-100]
            thresholding: Eşikleme yöntemi ("value" veya "connection")
        """
        super().__init__()
        
        self.num_windows = num_windows
        self.num_nodes = num_nodes
        self.sparsity = sparsity
        self.thresholding = thresholding
        
        self.correlation_builder = CorrelationMatrixBuilder(
            num_windows=num_windows,
            num_nodes=num_nodes
        )
    
    def forward(
        self,
        H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LSTM çıktısından grafik oluşturur.
        
        Args:
            H: LSTM çıktısı (batch, T, hidden_size)
            
        Returns:
            node_features: Düğüm öznitelikleri (batch, N, F)
            edge_index: Kenar indeksleri (2, num_edges) - batch için genişletilmiş
            edge_weight: Kenar ağırlıkları (num_edges,) - opsiyonel
        """
        # KBM hesapla
        node_features, C = self.correlation_builder(H)
        
        # Pencereler üzerinden ortalama al
        C_mean = C.mean(dim=1)  # (batch, N, N)
        
        # Eşikleme uygula
        if self.thresholding == "value":
            adjacency, edge_weight = self._value_thresholding(C_mean)
        else:
            adjacency, edge_weight = self._connection_thresholding(C_mean)
        
        # Edge index oluştur
        edge_index = self._adjacency_to_edge_index(adjacency)
        
        return node_features, edge_index, edge_weight
    
    def _value_thresholding(
        self,
        C: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Değer Eşikleme (DE) - Eşitlik 11.
        
        Kenar_ij = 1, eğer C_ij > percentile(C, 100-a)
        """
        batch_size, N, _ = C.shape
        
        # Percentile hesapla
        percentile = 100.0 - self.sparsity
        
        # Her örnek için eşik değeri
        C_flat = C.reshape(batch_size, -1)
        k = int(percentile / 100.0 * C_flat.shape[1])
        k = max(1, min(k, C_flat.shape[1] - 1))
        
        threshold, _ = torch.kthvalue(C_flat, k, dim=1)
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)
        
        # Eşikleme (ağırlıksız grafik)
        adjacency = (C > threshold).float()
        
        # Self-loop'ları kaldır
        adjacency = adjacency * (1 - torch.eye(N, device=C.device))
        
        return adjacency, None
    
    def _connection_thresholding(
        self,
        C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bağlantı Eşikleme (BE) - Eşitlik 12-13.
        
        Her düğüm için sabit sayıda en güçlü bağlantı korunur.
        """
        batch_size, N, _ = C.shape
        
        # Her düğüm için korunacak bağlantı sayısı
        n_connections = max(1, int(N * self.sparsity / 100.0))
        
        # Mutlak değer al
        C_abs = torch.abs(C)
        
        # Self-loop'ları sıfırla
        mask = 1 - torch.eye(N, device=C.device)
        C_abs = C_abs * mask
        
        # Her düğüm için top-k bağlantı
        _, indices = torch.topk(C_abs, k=n_connections, dim=-1)
        
        # Adjacency matrix oluştur
        adjacency = torch.zeros_like(C)
        
        for b in range(batch_size):
            for i in range(N):
                adjacency[b, i, indices[b, i]] = 1.0
        
        # Simetrik yap
        adjacency = torch.maximum(adjacency, adjacency.transpose(-2, -1))
        
        # Ağırlıklar (orijinal korelasyon değerleri)
        edge_weight = C * adjacency
        
        return adjacency, edge_weight
    
    def _adjacency_to_edge_index(
        self,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Adjacency matrix'i edge_index formatına dönüştürür.
        
        Args:
            adjacency: Adjacency matrix (batch, N, N)
            
        Returns:
            edge_index: (2, num_edges)
        """
        # İlk örneği kullan (batch içinde aynı yapı varsayılıyor)
        adj = adjacency[0]
        
        # Sıfır olmayan indeksleri bul
        edge_index = torch.nonzero(adj, as_tuple=False).t()
        
        return edge_index
