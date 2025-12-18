"""
EpiGraphNet Basitleştirilmiş Model
Makaledeki mimariye daha sadık, daha stabil implementasyon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Literal


class SimpleCNNLSTM(nn.Module):
    """Basitleştirilmiş CNN-LSTM modülü."""
    
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32, 64],
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # CNN katmanları
        layers = []
        in_ch = in_channels
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch
        
        self.cnn = nn.Sequential(*layers)
        
        # Adaptive pooling - sabit boyut çıktısı
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = lstm_hidden
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, length)
        Returns:
            H: (batch, seq_len, hidden) - LSTM tüm çıktıları
        """
        # CNN
        x = self.cnn(x)  # (batch, channels, length')
        
        # Adaptive pool
        x = self.adaptive_pool(x)  # (batch, channels, 64)
        
        # LSTM için: (batch, seq, features)
        x = x.permute(0, 2, 1)  # (batch, 64, channels)
        
        # LSTM
        H, _ = self.lstm(x)  # (batch, 64, hidden)
        H = self.dropout(H)
        
        return H


class SimpleGraphBuilder(nn.Module):
    """Basitleştirilmiş grafik oluşturucu."""
    
    def __init__(
        self,
        num_nodes: int = 16,
        sparsity: float = 25.0,
        thresholding: str = "value"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.sparsity = sparsity
        self.thresholding = thresholding
    
    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            H: LSTM çıktısı (batch, seq_len, hidden)
        Returns:
            node_features: (batch, N, F)
            adjacency: (batch, N, N)
        """
        batch_size, seq_len, hidden = H.shape
        
        # H'yi düzleştir ve düğümlere böl
        H_flat = H.reshape(batch_size, -1)  # (batch, seq_len * hidden)
        total_features = H_flat.shape[1]
        
        # N düğüme böl
        features_per_node = total_features // self.num_nodes
        if features_per_node == 0:
            features_per_node = 1
            # Padding ekle
            pad_size = self.num_nodes - total_features
            if pad_size > 0:
                H_flat = F.pad(H_flat, (0, pad_size))
        
        # Düğüm öznitelikleri: (batch, N, F)
        node_features = H_flat[:, :self.num_nodes * features_per_node]
        node_features = node_features.reshape(batch_size, self.num_nodes, -1)
        
        # Korelasyon matrisi hesapla
        # Normalize et
        node_norm = node_features - node_features.mean(dim=-1, keepdim=True)
        node_std = node_norm.std(dim=-1, keepdim=True) + 1e-8
        node_norm = node_norm / node_std
        
        # Korelasyon: (batch, N, N)
        correlation = torch.bmm(node_norm, node_norm.transpose(-2, -1))
        correlation = correlation / node_features.shape[-1]
        
        # Eşikleme
        if self.thresholding == "value":
            adjacency = self._value_threshold(correlation)
        else:
            adjacency = self._connection_threshold(correlation)
        
        return node_features, adjacency
    
    def _value_threshold(self, C: torch.Tensor) -> torch.Tensor:
        """Değer eşikleme (DE)."""
        batch_size, N, _ = C.shape
        
        # Percentile hesapla
        percentile = 100.0 - self.sparsity
        C_flat = C.reshape(batch_size, -1)
        k = max(1, int(percentile / 100.0 * C_flat.shape[1]))
        k = min(k, C_flat.shape[1] - 1)
        
        threshold, _ = torch.kthvalue(C_flat, k, dim=1)
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)
        
        adjacency = (C > threshold).float()
        
        # Self-loop'ları kaldır
        eye = torch.eye(N, device=C.device).unsqueeze(0)
        adjacency = adjacency * (1 - eye)
        
        return adjacency
    
    def _connection_threshold(self, C: torch.Tensor) -> torch.Tensor:
        """Bağlantı eşikleme (BE)."""
        batch_size, N, _ = C.shape
        n_connections = max(1, int(N * self.sparsity / 100.0))
        
        # Mutlak değer
        C_abs = torch.abs(C)
        
        # Self-loop'ları sıfırla
        eye = torch.eye(N, device=C.device).unsqueeze(0)
        C_abs = C_abs * (1 - eye)
        
        # Top-k
        _, indices = torch.topk(C_abs, k=min(n_connections, N-1), dim=-1)
        
        adjacency = torch.zeros_like(C)
        batch_idx = torch.arange(batch_size, device=C.device).view(-1, 1, 1)
        node_idx = torch.arange(N, device=C.device).view(1, -1, 1)
        adjacency.scatter_(2, indices, 1.0)
        
        # Simetrik yap
        adjacency = torch.maximum(adjacency, adjacency.transpose(-2, -1))
        adjacency = adjacency * (1 - eye)
        
        return adjacency


class SimpleGCN(nn.Module):
    """Basitleştirilmiş GCN modülü - Eşitlik 14."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # İlk katman
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        
        # Ara katmanlar
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_channels
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Eşitlik 14: G' = ReLU(W * (G + Σ G_neighbors))
        
        Args:
            x: Node features (batch, N, F)
            adj: Adjacency matrix (batch, N, N)
        Returns:
            x: Updated features (batch, hidden)
        """
        for layer in self.layers:
            # Mesaj toplama: her düğüm kendisi + komşuları
            # adj @ x = komşu toplamı
            neighbor_sum = torch.bmm(adj, x)
            combined = x + neighbor_sum
            
            # Linear + ReLU
            x = layer(combined)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global max pooling
        x = x.max(dim=1)[0]  # (batch, hidden)
        
        return x


class EpiGraphNetSimple(nn.Module):
    """
    Basitleştirilmiş EpiGraphNet modeli.
    Daha stabil eğitim için optimize edilmiş.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32, 64],
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        num_nodes: int = 16,
        sparsity: float = 25.0,
        thresholding: str = "value",
        gcn_hidden: int = 64,
        gcn_layers: int = 3,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # CNN-LSTM
        self.cnn_lstm = SimpleCNNLSTM(
            in_channels=in_channels,
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout
        )
        
        # Graph Builder
        self.graph_builder = SimpleGraphBuilder(
            num_nodes=num_nodes,
            sparsity=sparsity,
            thresholding=thresholding
        )
        
        # GCN - lazy init
        self.gcn = None
        self.gcn_hidden = gcn_hidden
        self.gcn_layers = gcn_layers
        self.dropout_rate = dropout
        
        # Classifier
        self.classifier = nn.Linear(gcn_hidden, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG signal (batch, 1, length)
        Returns:
            logits: (batch, num_classes)
        """
        # CNN-LSTM
        H = self.cnn_lstm(x)  # (batch, seq, hidden)
        
        # Graph oluştur
        node_features, adjacency = self.graph_builder(H)
        
        # GCN lazy init
        if self.gcn is None:
            self.gcn = SimpleGCN(
                in_channels=node_features.shape[-1],
                hidden_channels=self.gcn_hidden,
                num_layers=self.gcn_layers,
                dropout=self.dropout_rate
            ).to(x.device)
        
        # GCN
        graph_embedding = self.gcn(node_features, adjacency)  # (batch, hidden)
        
        # Classifier
        logits = self.classifier(graph_embedding)
        
        return logits
