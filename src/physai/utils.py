import torch

def to_device(tensor, device='cpu'):
    return tensor.to(device)
