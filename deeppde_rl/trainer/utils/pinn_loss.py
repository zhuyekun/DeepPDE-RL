import torch

def pinn_loss(x, y):
    return torch.mean(torch.square(x - y))
