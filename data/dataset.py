import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split

from .preprocessing import load_bonn_file, get_class_label, preprocess_eeg


class BonnEEGDataset(Dataset):
    def __init__(
        self,
        signals,
        labels,
        transform: Optional[callable] = None
    ):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label


def load_bonn_dataset(
    data_dir: str,
    binary: bool = False,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    signals = []
    labels = []
    valid_prefixes = ['Z', 'O', 'N', 'F', 'S']
    
    for filename in sorted(os.listdir(data_dir)):
        if not filename.lower().endswith('.txt'):
            continue
        
        prefix = filename[0].upper()
        if prefix not in valid_prefixes:
            continue
        
        filepath = os.path.join(data_dir, filename)
        
        try:
            signal = load_bonn_file(filepath)
            label = get_class_label(filename, binary=binary)
            
            signal = preprocess_eeg(signal, normalize=normalize, add_channel=True)
            
            signals.append(signal)
            labels.append(label)
            
        except Exception as e:
            pass
    
    return np.array(signals), np.array(labels)


def create_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    binary: bool = False,
    normalize: bool = True,
    random_seed: int = 42,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    signals, labels = load_bonn_dataset(data_dir, binary=binary, normalize=normalize)
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        signals, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=labels
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=random_seed,
        stratify=y_temp
    )
    
    train_dataset = BonnEEGDataset(X_train, y_train)
    val_dataset = BonnEEGDataset(X_val, y_val)
    test_dataset = BonnEEGDataset(X_test, y_test)
    
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    }
    return loaders
