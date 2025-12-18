"""
CNN-LSTM Birleşik Modül
Makaledeki Bölüm II.B - Hibrit CNN-LSTM mimarisi

Şekil 1'e göre akış:
EEG -> Conv1D -> MaxPool -> Conv1D -> Conv1D -> FC -> Dropout -> LSTM -> LSTM -> H

H tensörü: (batch, T, hidden_size) şeklinde çıkmalı
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from .cnn_module import CNNEncoder
from .lstm_module import LSTMEncoder


class CNNLSTM(nn.Module):
    """
    CNN-LSTM Hibrit Modül.
    
    CNN ile yerel zamansal öznitelikleri,
    LSTM ile uzun vadeli bağımlılıkları yakalar.
    
    Makaledeki Eşitlik 2-5'i uygular.
    Çıkış: H tensörü (batch, T, hidden_size)
    """
    
    def __init__(
        self,
        # CNN parametreleri
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [5, 5, 5],
        pool_size: int = 2,
        fc_hidden: int = 128,
        # LSTM parametreleri
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        bidirectional: bool = False,
        # Genel
        dropout: float = 0.1,
        # Sequence uzunluğu (T)
        sequence_length: int = 8
    ):
        """
        Args:
            in_channels: CNN giriş kanal sayısı
            conv_channels: CNN kanal sayıları
            kernel_sizes: CNN kernel boyutları
            pool_size: İlk katmandaki MaxPool boyutu
            fc_hidden: CNN FC katman boyutu
            lstm_hidden: LSTM gizli boyutu
            lstm_layers: LSTM katman sayısı
            bidirectional: Çift yönlü LSTM
            dropout: Dropout oranı
            sequence_length: LSTM için sequence uzunluğu (T)
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # CNN Encoder
        self.cnn = CNNEncoder(
            in_channels=in_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            pool_size=pool_size,
            fc_hidden=fc_hidden,
            dropout=dropout
        )
        
        # LSTM Encoder (2 katman - Şekil 1)
        self.lstm = LSTMEncoder(
            input_size=fc_hidden // sequence_length,  # Her zaman adımı için boyut
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Çıkış boyutu
        self.output_size = self.lstm.output_size
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: EEG sinyali (batch, 1, length)
            
        Returns:
            H: LSTM tüm gizli durumları (batch, T, hidden_size)
            (h_n, c_n): Son gizli durumlar
        """
        batch_size = x.shape[0]
        
        # CNN ile öznitelik çıkarımı
        # x: (batch, 1, length) -> z: (batch, fc_hidden)
        z = self.cnn(x)
        
        # Makaledeki H ∈ R^(s×T×h) tensörünü oluşturmak için
        # z'yi T zaman adımına böl: (batch, T, fc_hidden/T)
        z_seq = z.reshape(batch_size, self.sequence_length, -1)
        
        # LSTM ile temporal encoding
        # H: (batch, T, hidden_size)
        H, (h_n, c_n) = self.lstm(z_seq)
        
        return H, (h_n, c_n)
    
    def get_output_size(self) -> int:
        """Çıkış öznitelik boyutunu döndür."""
        return self.output_size
