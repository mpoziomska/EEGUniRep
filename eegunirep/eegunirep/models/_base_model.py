import torch.nn as nn

from eegunirep.utils.utils import get_logger


class BaseNet(nn.Module):
    model_name = 'CHANGE_ME'

    def __init__(self, device, dtype, loglevel):
        super().__init__()
        self.logger = get_logger(name="MODEL", loglevel=loglevel)
        self.device = device
        self.dtype = dtype
