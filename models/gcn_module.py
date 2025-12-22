import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
try:
    from torch_geometric.nn import global_max_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Uyarı: torch_geometric bulunamadı. Manuel global max pool kullanılacak.")


class ManualGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        neighbor_sum = torch.bmm(adjacency, x) 
        combined = x + neighbor_sum
        out = self.linear(combined)
        
        return out


class GCNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv = ManualGraphConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.conv(x, adjacency)
        x = F.relu(x)
        x = self.dropout(x)
        
        return x


class GCNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 5,
        num_layers: int = 3, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.use_pyg = HAS_PYG
        layers = []

        layers.append(GCNLayer(in_channels, hidden_channels, dropout))

        for _ in range(num_layers - 1):
            layers.append(GCNLayer(hidden_channels, hidden_channels, dropout))
        
        self.gcn_layers = nn.ModuleList(layers)

        self.fc = nn.Linear(hidden_channels, num_classes)
        self.final_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        batch: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        return self._forward_manual(x, adjacency)
    
    def _forward_manual(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:

        for layer in self.gcn_layers:
            x = layer(x, adjacency=adjacency)

        x = x.max(dim=1)[0]

        x = self.fc(x)
        x = self.final_dropout(x)
        
        return x
