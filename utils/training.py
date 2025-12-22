import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from .metrics import calculate_all_metrics


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch_y.cpu().tolist())
    
    avg_loss = total_loss / len(train_loader.dataset)
    metrics = calculate_all_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch_y.cpu().tolist())
    
    avg_loss = total_loss / len(val_loader.dataset)
    metrics = calculate_all_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    
    return metrics
