from abc import ABC, abstractmethod
import torch

class BaseLoader(ABC):
    def __init__(self, csv_path, label_col='label', num_worker=8, batch_size=64, device='cpu', dtype=torch.float32):
        self.label_col = label_col
        self.csv_path = csv_path
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.device = device
        self.dtype=dtype