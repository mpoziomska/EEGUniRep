from torcheeg import transforms as eeg_transforms

from torchvision import transforms as vision_transforms

from eegunirep.transforms import BaseTransforms
from eegunirep.utils.utils import get_debug_info
from eegunirep.utils.electrode_utils import CHANNEL_POSITION_MATRIX_idx, CHAN_LIST

import pandas as pd
from scipy.signal import sosfiltfilt, filtfilt
import numpy as np
import json
import gc
from numba import njit

class OnlineTransforms(BaseTransforms):
    def __init__(self, device, dtype, config, profiler, loglevel):
        super(OnlineTransforms, self).__init__(device, dtype, config, profiler=profiler, loglevel=loglevel)
        with open(config['transformations']['filters_pth'], 'r') as f:
            self.filter_params = json.load(f)

        self.len_crop = int(config['transformations']['len_crop_s'] * self.Fs)
        self.num_crops = config['transformations']['num_crops']
        self.len_sig_samp = self.len_crop * self.num_crops
        self.cut_marigin_samp = config['transformations']['cut_marigin_s'] * self.Fs
        self.sig_len_before_filtering = self.len_sig_samp + 2 * self.cut_marigin_samp
        self.get_stride_mask_limits(config=config)
        self.min_rect_len_perc = config['transformations']['min_rect_len_perc']
        self.max_rect_len_perc = config['transformations']['max_rect_len_perc']
        self.min_rect_amp = config['transformations']['min_rect_amp']
        self.max_rect_amp = config['transformations']['max_rect_amp']
        self.normalize = bool(config['transformations']['normalize'])
        self.logger.info(f"Normalize: {self.normalize}")
        self.noise_mean = config['transformations']['noise_mean']
        self.noise_std = config['transformations']['noise_std']
        self.noise_p = config['transformations']['noise_p']
        self.construct_eeg_transforms()
        self.construct_reference_transforms()
        self.transform = self.compose_transforms()

        self.transform_prime = self.compose_transforms()

    def get_stride_mask_limits(self, config):
        min_len_perc = config['transformations']['min_mask_len_perc']
        max_len_perc = config['transformations']['max_mask_len_perc']
        
        self.first_conv_stride = config['model']['feature_extractor']['temp_filter_length_inp']
        min_lim_samp = int(min_len_perc * self.len_crop) 
        max_lim_samp = int(max_len_perc * self.len_crop)

        min_lim_stride = self.first_conv_stride * (min_lim_samp // self.first_conv_stride)
        if min_lim_stride < min_lim_samp:
            min_lim_stride += self.first_conv_stride

        max_lim_stride = self.first_conv_stride * (max_lim_samp // self.first_conv_stride)
        if max_lim_stride > max_lim_samp:
            max_lim_stride += self.first_conv_stride
        
        self.stride_mask_limits = np.arange(min_lim_stride, max_lim_stride + 1, self.first_conv_stride)

    def __call__(self, sample):
        if self.profiler:
            self.profiler_handler.start()
        start_cut_idx = np.random.randint(low=0, high=sample.shape[-1] - self.sig_len_before_filtering)
        sample = sample[:, start_cut_idx: start_cut_idx + self.sig_len_before_filtering]
        x1 = self.transform(sample.copy())
        x2 = self.transform_prime(sample)
        if self.profiler:
            self.profiler_handler.stop()
            self.profiler_handler.print()
        return x1, x2
    
    def compose_transforms(self):
        return vision_transforms.Compose(
            [
                vision_transforms.RandomChoice(self.lambdas_reference, p=[0.2, 0.4, 0.4]),
                vision_transforms.RandomChoice([vision_transforms.Lambda(lambd=self.random_rect), vision_transforms.Lambda(lambd=lambda x: x)]),
                vision_transforms.Lambda(lambd=self.filter_sig),
                vision_transforms.RandomChoice([vision_transforms.Lambda(lambd=self.notch_60_filter), vision_transforms.Lambda(lambd=lambda x: x)]),
                vision_transforms.Lambda(lambd=self.crop_signal),
                vision_transforms.RandomChoice([vision_transforms.Lambda(lambd=self.make_random_masking), vision_transforms.Lambda(lambd=lambda x: x)]),
                vision_transforms.Lambda(lambd=lambda x: self.eeg_transforms(eeg=x)['eeg']),
            ]
        )
    
    def filter_sig(self, x):
        new_filter = np.random.choice(self.filter_params)
        # print(new_filter, x.dtype)
        x = sosfiltfilt(new_filter['lowpass'], x)
        x = sosfiltfilt(new_filter['highpass'], x)
        x = filtfilt(new_filter['notch_50']['b'], new_filter['notch_50']['a'], x, padlen=new_filter['notch_50']['padlen'])
        # self.logger.debug(f"filter_sig {get_debug_info(x)}")

        return x.copy()
    
    def notch_60_filter(self, x):
        new_filter = np.random.choice(self.filter_params)
        x = filtfilt(new_filter['notch_60']['b'], new_filter['notch_60']['a'], x, padlen=new_filter['notch_60']['padlen'])
        # self.logger.debug(f"notch_60_filter {get_debug_info(x)}")
        return x.copy()
    
    def construct_reference_transforms(self):
        self.lambdas_reference = [vision_transforms.Lambda(lambd=self.hjorth_rereference),
                                  vision_transforms.Lambda(lambd=self.average_rereference),
                                  vision_transforms.Lambda(lambd=self.median_rereference)]
    
    def construct_eeg_transforms(self):
        transforms_list = [
                eeg_transforms.ToTensor(),
                eeg_transforms.RandomNoise(mean=self.noise_mean, std=self.noise_std, p=self.noise_p),
            ]
        
        if self.normalize:
            transforms_list.append(eeg_transforms.MeanStdNormalize())
        self.eeg_transforms = eeg_transforms.Compose(transforms_list)

    @staticmethod
    @njit
    def numba_make_random_masking(X, stride_mask_limits, CHANNEL_POSITION_MATRIX_idx, idx_crops):
        # print(stride_mask_limits)
        for idx_crop in idx_crops:
            len_mask_idx = np.random.randint(low=0, high=len(stride_mask_limits)) # np.random.randint(int(self.min_mask_len_perc * self.len_crop), int(self.max_mask_len_perc * self.len_crop)) 
            start_mask_idx = np.random.randint(low=0, high=len(stride_mask_limits) - len_mask_idx) # np.random.randint(0, self.len_crop - len_mask)
            len_mask = stride_mask_limits[len_mask_idx]
            start_mask = stride_mask_limits[start_mask_idx]
            # print(start_mask, len_mask)
            idx0_pos = np.random.randint(CHANNEL_POSITION_MATRIX_idx.shape[0] - 1)
            idx1_pos = np.random.randint(CHANNEL_POSITION_MATRIX_idx.shape[1] - 1)
            picked_chans = CHANNEL_POSITION_MATRIX_idx[idx0_pos: idx0_pos + 2, idx1_pos: idx1_pos + 2].flatten()
            picked_chans = picked_chans[picked_chans >= 0]
            X[picked_chans, idx_crop, start_mask: start_mask + len_mask] = 0
        return X

    def make_random_masking(self, X):
        # maskowanie wycinków
        num_masked_crops = np.random.randint(self.num_crops)
        idx_crops = np.random.choice(np.arange(self.num_crops), size=num_masked_crops, replace=False)
        return self.numba_make_random_masking(X, self.stride_mask_limits, CHANNEL_POSITION_MATRIX_idx, idx_crops)

    def random_rect(self, X):
        num_rect_crops = np.random.randint(self.num_crops // 4)
        idx_crops = np.random.choice(np.arange(self.num_crops), size=num_rect_crops, replace=False)

        for idx_crop in idx_crops:
            len_mask = np.random.randint(int(self.min_rect_len_perc * self.len_crop), int(self.max_rect_len_perc * self.len_crop))
            start_mask = np.random.randint(0, self.len_crop - len_mask)
            idx_chan = np.random.randint(len(CHAN_LIST))
            X[idx_chan, int(self.len_crop * idx_crop) + start_mask: int(self.len_crop * idx_crop) + start_mask + len_mask] += (self.max_rect_amp - self.min_rect_amp) * np.random.random_sample() + self.min_rect_amp # (b - a) * random_sample() + a
        return X

    def crop_signal(self, X):
        X = X[:, self.cut_marigin_samp: -self.cut_marigin_samp]
        # self.logger.info("CROP")

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