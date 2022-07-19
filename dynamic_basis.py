import torch
import math


def dynamic_basis(input, current_epoch, total_epoch, num_basis):
    """
    Args:
        input:  (batch, num_basis, 3)
        current_epoch:
        total_epoch:
        num_basis:
    Returns:
    """
    alpha = current_epoch / total_epoch * num_basis
    k = torch.arange(num_basis, dtype=torch.float32, device=input.device)
    weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(math.pi).cos_()) / 2
    weight = weight[None, :, None]
    weighted_input = input * weight
    return weighted_input
