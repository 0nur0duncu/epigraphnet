"""
EpiGraphNet Ana Model
Makaledeki Şekil 1 - Genel mimari
CNN-LSTM + KBM + GCN entegrasyonu

NOT: Makaledeki Eşitlik 14'e göre GCN tek W matrisi kullanmalı.
PyTorch Geometric'in GraphConv'u iki matris kullandığı için,
her zaman ManualGraphConv ile adjacency matrix tabanlı çalışıyoruz.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal, List

from .cnn_lstm import CNNLSTM
from .graph_builder import GraphBuilder
from .gcn_module import GCNClassifier


class EpiGraphNet(nn.Module):
    """
    EpiGraphNet: Grafik Tabanlı EEG Epilepsi Tanı Modeli
    
    Bileşenler:
    1. CNN-LSTM: Yerel ve uzun vadeli zamansal öznitelik çıkarımı
    2. Graph Builder: KBM hesaplama ve grafik oluşturma
    3. GCN: Grafik düzeyinde sınıflandırma
    """
    
    def __init__(
        self,
        # CNN parametreleri
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [5, 5, 5],
        pool_size: int = 2,  # Şekil 1: sadece ilk katmanda pool
        fc_hidden: int = 128,
        # LSTM parametreleri
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        sequence_length: int = 8,  # T - zaman adımı sayısı
        # Grafik parametreleri
        num_windows: int = 8,
        num_nodes: int = 16,
        sparsity: float = 50.0,
        thresholding: Literal["value", "connection"] = "value",
        # GCN parametreleri
        gcn_hidden: int = 64,
        gcn_layers: int = 3,  # Şekil 1'de 3 GraphConv katmanı
        # Sınıflandırma
        num_classes: int = 5,
        # Genel
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: EEG giriş kanal sayısı
            conv_channels: CNN kanal sayıları
            kernel_sizes: CNN kernel boyutları
            pool_size: İlk katmandaki MaxPool boyutu (Şekil 1)
            fc_hidden: CNN FC katman boyutu
            lstm_hidden: LSTM gizli boyutu
            lstm_layers: LSTM katman sayısı
            sequence_length: LSTM için zaman adımı sayısı (T)
            num_windows: Grafik pencere sayısı (W)
            num_nodes: Grafik düğüm sayısı (N)
            sparsity: Seyreklik parametresi (a)
            thresholding: Eşikleme yöntemi ("value" veya "connection")
            gcn_hidden: GCN gizli kanal sayısı
            gcn_layers: GCN katman sayısı (Şekil 1'de 3)
            num_classes: Sınıf sayısı
            dropout: Dropout oranı
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        
        # 1. CNN-LSTM Modülü (Şekil 1'e göre)
        self.cnn_lstm = CNNLSTM(
            in_channels=in_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            pool_size=pool_size,
            fc_hidden=fc_hidden,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            sequence_length=sequence_length
        )
        
        # 2. Graph Builder
        self.graph_builder = GraphBuilder(
            num_windows=num_windows,
            num_nodes=num_nodes,
            sparsity=sparsity,
            thresholding=thresholding
        )
        
        # GCN giriş boyutunu hesapla (lazy initialization)
        self._gcn_in_channels = None
        self.gcn_hidden = gcn_hidden
        self.gcn_layers = gcn_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        # 3. GCN Classifier (lazy init)
        self.gcn = None
    
    def _init_gcn(self, in_channels: int):
        """GCN modülünü lazy olarak başlat."""
        self.gcn = GCNClassifier(
            in_channels=in_channels,
            hidden_channels=self.gcn_hidden,
            num_classes=self.num_classes,
            num_layers=self.gcn_layers,
            dropout=self.dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG sinyali (batch, 1, length)
            
        Returns:
            Sınıf logitleri (batch, num_classes)
        """
        batch_size = x.shape[0]
        
        # 1. CNN-LSTM ile öznitelik çıkarımı
        H, _ = self.cnn_lstm(x)  # H: (batch, T, hidden_size)
        
        # 2. Grafik oluşturma
        node_features, edge_index, edge_weight = self.graph_builder(H)
        # node_features: (batch, N, F)
        # edge_index: (2, num_edges)
        
        # GCN'i lazy başlat
        if self.gcn is None:
            self._init_gcn(node_features.shape[-1])
            self.gcn = self.gcn.to(x.device)
        
        # 3. GCN ile sınıflandırma (her zaman adjacency matrix tabanlı)
        # Makaleye uyumluluk için ManualGraphConv kullanıyoruz
        adjacency = self._edge_index_to_adjacency(
            edge_index, self.num_nodes, batch_size, x.device
        )
        logits = self.gcn(
            node_features, edge_index=None, adjacency=adjacency
        )
        
        return logits
    
    def _edge_index_to_adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Edge index'i adjacency matrix'e dönüştür."""
        adjacency = torch.zeros(
            batch_size, num_nodes, num_nodes,
            device=device
        )
        
        if edge_index.numel() > 0:
            row, col = edge_index[0], edge_index[1]
            for b in range(batch_size):
                adjacency[b, row, col] = 1.0
        
        return adjacency
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EpiGraphNet":
        """Konfigürasyondan model oluştur."""
        model_config = config.get("model", {})
        
        return cls(
            # CNN (Şekil 1'e göre)
            in_channels=model_config.get("cnn", {}).get("in_channels", 1),
            conv_channels=model_config.get("cnn", {}).get("conv_channels", [16, 32, 64]),
            kernel_sizes=model_config.get("cnn", {}).get("kernel_sizes", [5, 5, 5]),
            pool_size=model_config.get("cnn", {}).get("pool_size", 2),
            fc_hidden=model_config.get("fc_hidden", 128),
            # LSTM
            lstm_hidden=model_config.get("lstm", {}).get("hidden_size", 64),
            lstm_layers=model_config.get("lstm", {}).get("num_layers", 2),
            sequence_length=model_config.get("lstm", {}).get("sequence_length", 8),
            # Graph
            num_windows=model_config.get("graph", {}).get("num_windows", 8),
            num_nodes=model_config.get("graph", {}).get("num_nodes", 16),
            sparsity=model_config.get("graph", {}).get("sparsity", 50.0),
            thresholding=model_config.get("graph", {}).get("thresholding", "value"),
            # GCN (Şekil 1'de 3 katman)
            gcn_hidden=model_config.get("gcn", {}).get("hidden_channels", 64),
            gcn_layers=model_config.get("gcn", {}).get("num_layers", 3),
            # General
            num_classes=config.get("data", {}).get("num_classes", 5),
            dropout=model_config.get("dropout", 0.1)
        )
