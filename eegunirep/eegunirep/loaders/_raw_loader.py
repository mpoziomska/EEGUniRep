import torch

from eegunirep.loaders import BaseLoader
from eegunirep.transforms import OnlineStaticTransforms
from torcheeg import transforms as eeg_transforms
from torcheeg.datasets.module import CSVFolderDataset

from torchvision import transforms as vision_transforms

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class RawLoader(BaseLoader):

    def __init__(self, csv_path, config, device, loglevel, profiler=False, dtype=torch.float32, io_size=(1048576 * (2**6)), io_path=None, io_mode='pickle', label_col='label'):
        super().__init__(csv_path=csv_path, config=config, device=device, dtype=dtype, label_col=label_col, loglevel=loglevel)

        self.transforms = OnlineStaticTransforms(device=self.device, dtype=self.dtype, config=config, profiler=profiler, loglevel=loglevel)

        self.dataset = CSVFolderDataset(config=config,
                                        csv_path=self.csv_path,
                                        online_transform=self.transforms,
                                        label_transform=eeg_transforms.Select(self.label_col),
                                        num_worker=self.num_worker,
                                        io_mode=io_mode,
                                        verbose=True,
                                        io_size=io_size,
                                        io_path=io_path,
                                        )
        
        self.loader = torch.utils.data.DataLoader(
                            self.dataset,
                            batch_size=config['training']['eval_batch_size'],
                            num_workers=self.num_worker,
                            pin_memory=True,
                            drop_last=True,
                            persistent_workers=False
                        )