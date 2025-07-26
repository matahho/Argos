from collections import defaultdict
import random
from torch import mps , cuda
import torch

NUMBER_OF_CLIENTS = 5

def device_allocation():
    if mps.is_available():
        return torch.device("mps")
    elif cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


