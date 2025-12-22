import torch
import torch.nn as nn
from typing import List, Optional


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        use_pool: bool = True,
        pool_size: int = 2,
        padding: str = "same"
    ):
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
        x = self.conv(x)
        x = self.relu(x)
        if self.use_pool:
            x = self.pool(x)
        x = self.bn(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [5, 5, 5],
        pool_size: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        blocks = []
        channels = [in_channels] + conv_channels
        
        for i in range(len(conv_channels)):
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
        self.fc = None
        self.fc_hidden = fc_hidden
        self.dropout = nn.Dropout(dropout)
        self._initialized = False
        
        self.out_channels = conv_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_blocks(x)
        
        x = self.flatten(x)

        if not self._initialized:
            flat_size = x.shape[1]
            self.fc = nn.Linear(flat_size, self.fc_hidden).to(x.device)
            self._initialized = True
        
        x = self.fc(x)
        
        x = self.dropout(x)
        
        return x
