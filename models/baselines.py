import torch
import torch.nn as nn
from typing import List


class BaselineCNNLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [5, 5, 5],
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        cnn_layers = []
        in_ch = in_channels
        for out_ch, ks in zip(conv_channels, kernel_sizes):
            cnn_layers.append(nn.Conv1d(in_ch, out_ch, ks, padding=ks//2))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(2))
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Baseline1DCNNLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64, 128, 64],
        kernel_sizes: List[int] = [7, 5, 5, 3],
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        cnn_layers = []
        in_ch = in_channels
        for out_ch, ks in zip(conv_channels, kernel_sizes):
            cnn_layers.append(nn.Conv1d(in_ch, out_ch, ks, padding=ks//2))
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(2))
            cnn_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x
