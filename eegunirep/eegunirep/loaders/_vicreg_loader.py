import torch

from eegunirep.loaders import BaseLoader
from eegunirep.augmentations import SimpleAugmentation
from eegunirep.datasets import VicRegDataset
from torcheeg import transforms as eeg_transforms

from torchvision import transforms as vision_transforms

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



class VicRegLoader(BaseLoader):

    def __init__(self, csv_path, label_col='label', num_worker=8, io_mode='pickle', io_size=1048576, io_path=None, batch_size=64, device='cpu', dtype=torch.float32, make_transform=True):
        super().__init__(csv_path, label_col, num_worker, batch_size, device, dtype)

        self.transforms = SimpleAugmentation(device=self.device, dtype=self.dtype) if make_transform else eeg_transforms.ToTensor()

        self.dataset = VicRegDataset(csv_path=self.csv_path,
                           online_transform=self.transforms,
                           label_transform=eeg_transforms.Select(self.label_col),
                           num_worker=self.num_worker,
                           io_mode=io_mode,
                           verbose=True,
                           io_size=io_size,
                           io_path=io_path)
        
        self.sampler = torch.utils.data.RandomSampler(data_source=self.dataset)

        self.loader =  torch.utils.data.DataLoader(
                            self.dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_worker,
                            pin_memory=True,
                            sampler=self.sampler,
                            drop_last=True,
                            persistent_workers=True
                        )
        
    def get_batched_loader(self, batch_size=64):
        pass

