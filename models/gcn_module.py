"""
GCN Modülü
Makaledeki Bölüm II.D - GCN ile Grafik Düzeyinde Sınıflandırma
Eşitlik 14-16: GraphConv, global_max_pool, softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# PyTorch Geometric import
try:
    from torch_geometric.nn import GraphConv, global_max_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Uyarı: torch_geometric bulunamadı. Manuel GCN kullanılacak.")


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
    GraphConv + ReLU + Dropout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if HAS_PYG:
            self.conv = GraphConv(in_channels, out_channels)
        else:
            self.conv = ManualGraphConv(in_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.use_pyg = HAS_PYG
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features
            edge_index: Edge indices (for PyG)
            adjacency: Adjacency matrix (for manual impl)
            
        Returns:
            Updated node features
        """
        if self.use_pyg:
            x = self.conv(x, edge_index)
        else:
            x = self.conv(x, adjacency)
        
        x = F.relu(x)
        x = self.dropout(x)
        
        return x


class GCNClassifier(nn.Module):
    """
    GCN tabanlı grafik sınıflandırıcı.
    
    Şekil 1'e göre: GraphConv x3 -> Global Max Pool -> FC -> Dropout -> Softmax
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
        
        self.use_pyg = HAS_PYG
        
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
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (N_total, F) veya (batch, N, F)
            edge_index: Edge indices (2, E)
            batch: Batch atamaları (N_total,) - PyG için
            adjacency: Adjacency matrix (batch, N, N) - manuel için
            
        Returns:
            Sınıf logitleri (batch_size, num_classes)
        """
        if self.use_pyg:
            return self._forward_pyg(x, edge_index, batch)
        else:
            return self._forward_manual(x, adjacency)
    
    def _forward_pyg(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch Geometric ile forward."""
        # GCN katmanları
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
        
        # Eşitlik 15: Global max pooling
        x = global_max_pool(x, batch)  # (batch_size, hidden_channels)
        
        # Eşitlik 16: FC -> Dropout (softmax CrossEntropyLoss içinde)
        x = self.fc(x)
        x = self.final_dropout(x)
        
        return x
    
    def _forward_manual(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Manuel implementasyon ile forward."""
        # x: (batch, N, F)
        
        # GCN katmanları
        for layer in self.gcn_layers:
            x = layer(x, edge_index=None, adjacency=adjacency)
        
        # Eşitlik 15: Global max pooling (düğümler üzerinden)
        x = x.max(dim=1)[0]  # (batch, hidden_channels)
        
        # Eşitlik 16: FC -> Dropout (softmax CrossEntropyLoss içinde)
        x = self.fc(x)
        x = self.final_dropout(x)
        
        return x
