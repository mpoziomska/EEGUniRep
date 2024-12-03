from abc import ABC, abstractmethod
import numpy as np
import mne

from eegunirep.utils.electrode_utils import CHANNEL_POSITION_MATRIX, CHAN_LIST

class BaseAugmentation(ABC):
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.Fs = 256
        self.len_crop = int(6 * self.Fs)
        self.num_crops = 30
        self.start_cut = int(1 * self.Fs)
        self.stop_cut = int(self.start_cut + self.num_crops *  self.len_crop)

    def __call__(self, sample):
        x1 = self.transform(sample.copy())
        x2 = self.transform_prime(sample)
        return x1, x2

    def cut_signal(self, X):
        return X[:, self.start_cut: self.stop_cut].reshape(X.shape[0], self.num_crops, self.len_crop)

    @staticmethod
    def average_rereference(data):
        print("AVERAGE REF")

        reference = -np.sum(data, axis=0) / (data.shape[0]+1)
        data += reference
        return data

    @staticmethod
    def median_rereference(data):
        print("MEDIAN REF")

        reference = -np.nanmedian(data, axis=0) * (
                data.shape[0] / (data.shape[0] + 1)
        )
        data += reference
        return data


    def hjorth_rereference(self, data):
        print("HJORTH REF")
        chan_pos = CHANNEL_POSITION_MATRIX
        edf = mne.io.RawArray(data=data, info=mne.create_info(ch_names=CHAN_LIST, sfreq=self.Fs))

        def make_ref(A, **kwargs):
            ref = np.mean(edf.copy().pick_channels(kwargs["ref"]).get_data(), axis=0)
            return A - ref

        for i in range(chan_pos.shape[0]):
            for j in range(chan_pos.shape[1]):
                if chan_pos[i, j] != "":
                    neigh = [
                        chan_pos[i, max(0, j - 1)],
                        chan_pos[min(chan_pos.shape[0] - 1, i + 1), j],
                        chan_pos[i, min(chan_pos.shape[1] - 1, j + 1)],
                        chan_pos[max(0, i - 1), j],
                    ]
                    while "" in neigh:
                        neigh.remove("")

                    if chan_pos[i, j] in neigh:
                        neigh.remove(chan_pos[i, j])
                    edf.apply_function(make_ref, picks=[chan_pos[i, j]], ref=neigh)
        return edf._data