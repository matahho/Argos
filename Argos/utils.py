from torch import mps , cuda
import torch


def device_allocation():
    if mps.is_available():
        return torch.device("mps")
    elif cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


