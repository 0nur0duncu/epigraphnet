"""
1D CNN Modülü
Makaledeki Bölüm II.B - CNN bileşeni
Eşitlik 2: x^(1) = BatchNorm(MaxPool(ReLU(Conv1D(x; W_1, b_1))))
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CNNBlock(nn.Module):
    """
    Tek bir CNN bloğu.
    Şekil 1'e göre: Conv1D -> ReLU -> (opsiyonel MaxPool) -> BatchNorm
    NOT: Şekil 1'de sadece ilk Conv'dan sonra MaxPool var!
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        use_pool: bool = True,
        pool_size: int = 2,
        padding: str = "same"
    ):
        """
        Args:
            in_channels: Giriş kanal sayısı
            out_channels: Çıkış kanal sayısı
            kernel_size: Konvolüsyon kernel boyutu
            use_pool: MaxPool kullan mı (Şekil 1'e göre sadece ilk katmanda)
            pool_size: MaxPool boyutu
            padding: Padding türü
        """
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Giriş tensörü (batch, channels, length)
            
        Returns:
            Çıkış tensörü
        """
        x = self.conv(x)
        x = self.relu(x)
        if self.use_pool:
            x = self.pool(x)
        x = self.bn(x)
        return x


class CNNEncoder(nn.Module):
    """
    Çok katmanlı 1D CNN Encoder.
    Birden fazla CNNBlock'u sıralı olarak uygular.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [5, 5, 5],
        pool_size: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Giriş kanal sayısı
            conv_channels: Her katmandaki kanal sayıları
            kernel_sizes: Her katmandaki kernel boyutları
            pool_size: İlk katmandaki MaxPool boyutu (Şekil 1)
            fc_hidden: FC katman gizli boyutu
            dropout: Dropout oranı
        """
        super().__init__()
        
        # CNN blokları oluştur
        # Şekil 1'e göre: Conv1D -> MaxPool -> Conv1D -> Conv1D
        # Sadece ilk katmanda MaxPool var!
        blocks = []
        channels = [in_channels] + conv_channels
        
        for i in range(len(conv_channels)):
            # Şekil 1'e göre: sadece ilk katmanda MaxPool
            use_pool = (i == 0)
            
            blocks.append(CNNBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                use_pool=use_pool,
                pool_size=pool_size
            ))
        
        self.cnn_blocks = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        
        # FC katman - boyut lazy olarak hesaplanacak
        self.fc = None
        self.fc_hidden = fc_hidden
        self.dropout = nn.Dropout(dropout)
        self._initialized = False
        
        # Çıkış kanal sayısı
        self.out_channels = conv_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Giriş tensörü (batch, 1, length)
            
        Returns:
            Öznitelik vektörü (batch, fc_hidden)
        """
        # CNN blokları
        x = self.cnn_blocks(x)
        
        # Flatten
        x = self.flatten(x)
        
        # FC katmanını lazy başlat
        if not self._initialized:
            flat_size = x.shape[1]
            self.fc = nn.Linear(flat_size, self.fc_hidden).to(x.device)
            self._initialized = True
        
        # FC + Dropout
        x = self.fc(x)
        x = self.dropout(x)
        
        return x
