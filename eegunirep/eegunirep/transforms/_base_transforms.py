from abc import ABC, abstractmethod
import numpy as np
from pyinstrument import Profiler

from eegunirep.utils.utils import get_logger
from eegunirep.utils.electrode_utils import CHANNEL_POSITION_MATRIX_idx, CHAN_LIST
from numba import njit


class BaseTransforms(ABC):
    def __init__(self, device, dtype, config, profiler, loglevel):
        self.device = device
        self.dtype = dtype
        self.Fs = config['preprocess']['new_Fs']
        self.profiler = profiler
        if self.profiler:
            self.profiler_handler = Profiler()
        self.logger = get_logger(name="TRANSFORMS", loglevel=loglevel)

    
    def crop_signal(self, X):
        # self.logger.info(f"CROP1: {X.shape}")

        X = X[:, self.cut_marigin_samp: -self.cut_marigin_samp]
        # self.logger.info(f"CROP2: {X.shape}")

        return X.reshape(X.shape[0], self.num_crops, self.len_crop)

    @staticmethod
    @njit
    def average_rereference(data):
        data -= np.sum(data, axis=0) / (data.shape[0]+1)
        return data

    @staticmethod
    def median_rereference(data):
        data -= np.median(a=data, axis=0) * (data.shape[0] / (data.shape[0] + 1))
        return data

    @staticmethod
    @njit
    def make_hjorth(data, chan_pos):
        data_copy = data.copy()
        for i in range(chan_pos.shape[0]):
            for j in range(chan_pos.shape[1]):
                main_idx = chan_pos[i, j]
                if main_idx != -1:
                    neigh = np.array([
                        chan_pos[i, max(0, j - 1)],
                        chan_pos[min(chan_pos.shape[0] - 1, i + 1), j],
                        chan_pos[i, min(chan_pos.shape[1] - 1, j + 1)],
                        chan_pos[max(0, i - 1), j],
                    ])
                    neigh = neigh[neigh >= 0]
                    neigh = neigh[neigh != main_idx]
                    data_neigh = data[neigh]
                    sum_val = np.zeros((data_neigh.shape[1]))
                    for i in range(data_neigh.shape[0]):
                        sum_val += data_neigh[i]
                    sum_val /= data_neigh.shape[0]
                    data_copy[main_idx] -= sum_val
        return data_copy
    
    def hjorth_rereference(self, data):
        new_data = self.make_hjorth(data, CHANNEL_POSITION_MATRIX_idx)
        # self.logger.debug(f"hjorth_rereference {get_debug_info(new_data)}")
        return new_data