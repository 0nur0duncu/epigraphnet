"""
GCN Modülü
Makaledeki Bölüm II.D - GCN ile Grafik Düzeyinde Sınıflandırma
Eşitlik 14-16: GraphConv, global_max_pool, softmax

NOT: Makale Eşitlik 14'e göre TEK W matrisi kullanılmalı:
G_i^(l+1) = ReLU(W^(l) * (G_i^(l) + Σ G_j^(l)))

PyTorch Geometric'in GraphConv'u İKİ matris kullanıyor (lin_rel + lin_root),
bu yüzden makaleye uyumluluk için her zaman ManualGraphConv kullanıyoruz.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# PyTorch Geometric import (sadece global_max_pool için)
try:
    from torch_geometric.nn import global_max_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Uyarı: torch_geometric bulunamadı. Manuel global max pool kullanılacak.")


class ManualGraphConv(nn.Module):
    """
    Manuel GraphConv implementasyonu.
    torch_geometric yoksa kullanılır.
    
    Eşitlik 14: G_i^(l+1) = ReLU(W^(l) * (G_i^(l) + Σ G_j^(l)))
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, N, in_channels)
            adjacency: Adjacency matrix (batch, N, N)
            
        Returns:
            Updated node features (batch, N, out_channels)
        """
        # Komşu mesajlarını topla
        neighbor_sum = torch.bmm(adjacency, x)  # (batch, N, in_channels)
        
        # Kendi özellikleriyle birleştir
        combined = x + neighbor_sum
        
        # Lineer dönüşüm
        out = self.linear(combined)
        
        return out


class GCNLayer(nn.Module):
    """
    Tek bir GCN katmanı.
    Makaledeki Eşitlik 14'e uygun: ManualGraphConv + ReLU + Dropout
    
    NOT: PyTorch Geometric'in GraphConv'u iki W matrisi kullanıyor,
    ancak makale TEK W matrisi gerektiriyor. Bu yüzden her zaman
    ManualGraphConv kullanıyoruz.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Makaleye uyumluluk için HER ZAMAN ManualGraphConv kullan
        # (PyTorch Geometric'in GraphConv'u 2 matris kullanıyor, makale 1 matris istiyor)
        self.conv = ManualGraphConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, N, in_channels)
            edge_index: Kullanılmıyor (PyG uyumluluğu için)
            adjacency: Adjacency matrix (batch, N, N)
            
        Returns:
            Updated node features (batch, N, out_channels)
        """
        x = self.conv(x, adjacency)
        x = F.relu(x)
        x = self.dropout(x)
        
        return x


class GCNClassifier(nn.Module):
    """
    GCN tabanlı grafik sınıflandırıcı.
    
    Makaleye göre (Şekil 1 ve Eşitlik 14-16):
    GraphConv x3 -> Global Max Pool -> FC -> Dropout -> Softmax
    
    NOT: Makaledeki Eşitlik 14 tek W matrisi gerektirdiği için
    PyTorch Geometric'in GraphConv'u yerine ManualGraphConv kullanıyoruz.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 5,
        num_layers: int = 3,  # Şekil 1'de 3 GraphConv katmanı var
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Giriş öznitelik boyutu
            hidden_channels: GCN gizli kanal sayısı
            num_classes: Sınıf sayısı
            num_layers: GCN katman sayısı (Şekil 1'de 3)
            dropout: Dropout oranı
        """
        super().__init__()
        
        self.use_pyg = HAS_PYG  # Sadece global_max_pool için
        
        # GCN katmanları (Şekil 1'de 3 adet)
        layers = []
        
        # İlk katman
        layers.append(GCNLayer(in_channels, hidden_channels, dropout))
        
        # Ara katmanlar
        for _ in range(num_layers - 1):
            layers.append(GCNLayer(hidden_channels, hidden_channels, dropout))
        
        self.gcn_layers = nn.ModuleList(layers)
        
        # Şekil 1'e göre: FC -> Dropout -> Softmax
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.final_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        batch: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, N, F) 
            edge_index: Kullanılmıyor (uyumluluk için)
            batch: Kullanılmıyor (uyumluluk için)
            adjacency: Adjacency matrix (batch, N, N)
            
        Returns:
            Sınıf logitleri (batch_size, num_classes)
        """
        # Her zaman manuel implementasyonu kullan (makaleye uyumluluk)
        return self._forward_manual(x, adjacency)
    
    def _forward_manual(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Manuel implementasyon ile forward (Eşitlik 14-16)."""
        # x: (batch, N, F)
        
        # Eşitlik 14: GCN katmanları
        for layer in self.gcn_layers:
            x = layer(x, adjacency=adjacency)
        
        # Eşitlik 15: Global max pooling (düğümler üzerinden)
        x = x.max(dim=1)[0]  # (batch, hidden_channels)
        
        # Eşitlik 16: FC -> Dropout (softmax CrossEntropyLoss içinde)
        x = self.fc(x)
        x = self.final_dropout(x)
        
        return x
