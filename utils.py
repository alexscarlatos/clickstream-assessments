import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_seeds(seedNum):
    random.seed(seedNum)
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seedNum)
        torch.cuda.manual_seed_all(seedNum)
