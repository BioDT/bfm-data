# src/data_preprocessing/transformation/text.py

import torch


def standarize_data(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    return (tensor - mean) / std


def normalise_data(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min(dim=0, keepdim=True).values
    max_val = tensor.max(dim=0, keepdim=True).values
    return (tensor - min_val) / (max_val - min_val)
