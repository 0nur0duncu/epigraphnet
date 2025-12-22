import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from .cnn_module import CNNEncoder
from .lstm_module import LSTMEncoder


class CNNLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [5, 5, 5],
        pool_size: int = 2,
        fc_hidden: int = 128,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.1,
        sequence_length: int = 8
    ):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        self.cnn = CNNEncoder(
            in_channels=in_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            pool_size=pool_size,
            fc_hidden=fc_hidden,
            dropout=dropout
        )
        
        self.lstm = LSTMEncoder(
            input_size=fc_hidden // sequence_length,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.output_size = self.lstm.output_size
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.shape[0]
        z = self.cnn(x)
        z_seq = z.reshape(batch_size, self.sequence_length, -1)
        H, (h_n, c_n) = self.lstm(z_seq)
        
        return H, (h_n, c_n)
    
    def get_output_size(self):
        return self.output_size
