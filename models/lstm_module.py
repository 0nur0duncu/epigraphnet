"""
LSTM Modülü
Makaledeki Bölüm II.B - LSTM bileşeni
Eşitlik 5: H_t, (h_t, c_t) = LSTM(z_drop, t=1,...,T)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMEncoder(nn.Module):
    """
    İki katmanlı LSTM Encoder.
    Uzun vadeli zamansal bağımlılıkları yakalar.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        batch_first: bool = True
    ):
        """
        Args:
            input_size: Giriş öznitelik boyutu
            hidden_size: LSTM gizli durum boyutu
            num_layers: LSTM katman sayısı
            dropout: Katmanlar arası dropout
            bidirectional: Çift yönlü LSTM
            batch_first: Batch boyutu ilk sırada mı
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        
        # Çıkış boyutu
        self.output_size = hidden_size * self.num_directions
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Giriş tensörü (batch, seq_len, input_size)
            hidden: Başlangıç gizli durumu (opsiyonel)
            
        Returns:
            H: Tüm gizli durumlar (batch, seq_len, hidden_size * num_directions)
            (h_n, c_n): Son gizli ve hücre durumları
        """
        # Eğer x 2D ise, sequence boyutu ekle
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        # LSTM forward
        H, (h_n, c_n) = self.lstm(x, hidden)
        
        return H, (h_n, c_n)
    
    def get_output_size(self) -> int:
        """LSTM çıkış boyutunu döndür."""
        return self.output_size
