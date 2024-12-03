from torcheeg import transforms as eeg_transforms
from torchaudio.functional import filtfilt

from torchaudio import transforms as audio_transforms

from torchvision.transforms import v2 as vision_transforms

from eegunirep.augmentations import BaseAugmentation
import pandas as pd
import mne
from scipy.signal import sosfiltfilt, filtfilt
import numpy as np

import json
import torch 


class SimpleAugmentation(BaseAugmentation):
    def __init__(self, device, dtype):
        super(SimpleAugmentation, self).__init__(device, dtype)
        with open("/dmj/fizmed/mpoziomska/ELMIKO/DOKTORAT/EEGUniRep/notebooks/filters.json", 'r') as f:
            self.filter_params = json.load(f)
        self.df_masking = pd.read_csv("/dmj/fizmed/mpoziomska/ELMIKO/DOKTORAT/EEGUniRep/notebooks/chans.csv")
        self.Fs = 256
        self.construct_filter_transforms()
        self.construct_eeg_transforms()
        self.construct_masking_transforms()
        self.construct_reference_transforms()
        self.transform = self.compose_transforms()

        self.transform_prime = self.compose_transforms()

    def compose_transforms(self):
        return vision_transforms.Compose(
            [
                vision_transforms.Lambda(lambd=self.filter_sig),
                vision_transforms.RandomChoice([vision_transforms.RandomChoice([vision_transforms.Lambda(lambd=self.make_random_masking), vision_transforms.Lambda(lambd=lambda x: x)])]),
                vision_transforms.RandomChoice(self.lambdas_reference),
                vision_transforms.Lambda(lambd=self.cut_signal),
                vision_transforms.Lambda(lambd=lambda x: self.eeg_transforms(eeg=x)['eeg']),
            ]
        )

    def filter_sig(self, x):
        new_filter = self.get_filters(1)
        edf_filt = sosfiltfilt(new_filter['lowpass'], x)
        edf_filt = sosfiltfilt(new_filter['highpass'], edf_filt)
        edf_filt = filtfilt(new_filter['notch']['b'], new_filter['notch']['a'], edf_filt, padlen=new_filter['notch']['padlen'])
        return edf_filt.copy()
        # return audio_filtfilt(x, a_coeffs = torch.tensor(params['a']).to(device=self.device, dtype=self.dtype), b_coeffs = torch.tensor(params['b']).to(device=self.device, dtype=self.dtype), clamp=False)
    
    def construct_filter_transforms(self):
        lambda_filters = []
        for filter in self.filter_params:
            lambda_filters.append(vision_transforms.Lambda(lambd=lambda x, filter=filter: filter))
        self.get_filters = vision_transforms.Compose([vision_transforms.RandomChoice(lambda_filters)]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    def construct_masking_transforms(self):

        sites = self.df_masking.site.unique()
        lambda_sites = []
        for site in sites:
            lambda_sites.append(vision_transforms.Lambda(lambd=lambda x, site=site: site))
        
        self.get_site = vision_transforms.Compose([vision_transforms.RandomChoice(lambda_sites)])

        areas = self.df_masking.columns
        lambda_areas = []
        for area in areas:
            if area not in ['chan', 'idx_chan', 'site']:
                lambda_areas.append(vision_transforms.Lambda(lambd=lambda x, area=area: area))
        self.get_area = vision_transforms.Compose([vision_transforms.RandomChoice(lambda_areas)])


        min_mask_len = 0.1 * self.len_crop
        max_mask_len = int(0.4 * self.num_crops) * self.len_crop
        start_points = np.arange(0, self.num_crops * self.len_crop - max_mask_len, 5)
        lambda_start_point = []
        for start_poit in start_points:
            lambda_start_point.append(vision_transforms.Lambda(lambd=lambda x, start_poit=start_poit: start_poit))
        self.get_start_point = vision_transforms.Compose([vision_transforms.RandomChoice(lambda_start_point)])

        mask_lens = np.arange(min_mask_len, max_mask_len, 5)
        lambda_mask_len = []
        for mask_len in mask_lens:
            lambda_mask_len.append(vision_transforms.Lambda(lambd=lambda x, mask_len=mask_len: mask_len))
        self.get_mask_len = vision_transforms.Compose([vision_transforms.RandomChoice(lambda_mask_len)])


    def make_random_masking(self, X):
        site = self.get_site(1)
        area = self.get_area(1)
        start_point = int(self.get_start_point(1))
        mask_len = int(self.get_mask_len(1))

        idx_chan = self.df_masking.idx_chan[(self.df_masking.site == site) * self.df_masking[area]]
        print(site, area, start_point, mask_len)
        X[idx_chan, start_point: start_point + mask_len] = 0

        return X
    
    def construct_reference_transforms(self):
        self.lambdas_reference = [vision_transforms.Lambda(lambd=self.hjorth_rereference),
                                  vision_transforms.Lambda(lambd=self.average_rereference),
                                  vision_transforms.Lambda(lambd=self.median_rereference)]
    
    def construct_eeg_transforms(self):

        self.eeg_transforms = eeg_transforms.Compose(
            [
                eeg_transforms.ToTensor(),
                eeg_transforms.RandomNoise(mean=0, std=4, p=0.2),
                eeg_transforms.RandomMask(ratio=0.2, p=0.2),
            ]
        )

   
    