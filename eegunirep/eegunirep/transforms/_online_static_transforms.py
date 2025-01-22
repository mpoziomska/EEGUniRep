from torcheeg import transforms as eeg_transforms

from torchvision import transforms as vision_transforms

from eegunirep.transforms import BaseTransforms
from eegunirep.utils.utils import get_debug_info

import pandas as pd
from scipy.signal import sosfiltfilt, filtfilt
import numpy as np
import json
import gc
from numba import njit


class OnlineStaticTransforms(BaseTransforms):
    def __init__(self, device, dtype, config, profiler, loglevel):
        super(OnlineStaticTransforms, self).__init__(device, dtype, config, profiler=profiler, loglevel=loglevel)
        with open(config['transformations']['filters_pth'], 'r') as f:
            self.filter_params = json.load(f)

        self.len_crop = int(config['transformations']['len_crop_s'] * self.Fs)
        self.num_crops = config['transformations']['num_crops']
        self.static_start_cut = config['transformations']['static_start_cut']
        self.len_sig_samp = self.len_crop * self.num_crops
        self.cut_marigin_samp = config['transformations']['cut_marigin_s'] * self.Fs
        self.sig_len_before_filtering = self.len_sig_samp + 2 * self.cut_marigin_samp
        self.normalize = bool(config['transformations']['normalize'])
        self.logger.info(f"Normalize: {self.normalize}")
        self.construct_eeg_transforms()

        self.transform = self.compose_transforms()

    def __call__(self, sample):
        if self.profiler:
            self.profiler_handler.start()
        # start_cut_idx = int(self.static_start_cut * self.Fs)
        start_cut_idx = np.random.randint(low=0, high=sample.shape[-1] - self.sig_len_before_filtering)

        sample = sample[:, start_cut_idx: start_cut_idx + self.sig_len_before_filtering]
        x = self.transform(sample.copy())
        if self.profiler:
            self.profiler_handler.stop()
            self.profiler_handler.print()
        return x
    
    def compose_transforms(self):
        return vision_transforms.Compose(
            [
                vision_transforms.Lambda(lambd = self.average_rereference),
                vision_transforms.Lambda(lambd=self.filter_sig),
                vision_transforms.Lambda(lambd=self.crop_signal),
                vision_transforms.Lambda(lambd=lambda x: self.eeg_transforms(eeg=x)['eeg']),
            ]
        )
    
    def filter_sig(self, x):
        new_filter = self.filter_params[0]
        # print(new_filter, x.dtype)
        x = sosfiltfilt(new_filter['lowpass'], x)
        x = sosfiltfilt(new_filter['highpass'], x)
        x = filtfilt(new_filter['notch_50']['b'], new_filter['notch_50']['a'], x, padlen=new_filter['notch_50']['padlen'])
        # self.logger.debug(f"filter_sig {get_debug_info(x)}")

        return x.copy()
    
    def construct_eeg_transforms(self):
        transforms_list = [
                eeg_transforms.ToTensor()
            ]
        
        if self.normalize:
            transforms_list.append(eeg_transforms.MeanStdNormalize())
        self.eeg_transforms = eeg_transforms.Compose(transforms_list)
        