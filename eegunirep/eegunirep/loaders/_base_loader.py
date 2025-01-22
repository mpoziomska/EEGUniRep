from abc import ABC, abstractmethod
import torch
from eegunirep.utils.utils import get_logger
from eegunirep.utils.electrode_utils import CHAN_LIST, CHNAMES_MAPPING, apply_mor_data_hack_fix
from torcheeg.model_selection import KFoldCrossTrial


class BaseLoader(ABC):
    def __init__(self, csv_path, config, device, loglevel, profiler=False, dtype=torch.float32, label_col='label'):
        self.label_col = label_col
        self.csv_path = csv_path
        self.num_worker = config['general']['workers']
        self.train_batch_size = config['training']['train_batch_size']
        self.eval_batch_size = config['training']['eval_batch_size']
        self.device = device
        self.dtype=dtype
        self.profiler = profiler
        self.logger = get_logger(name="LOADER", loglevel=loglevel)
        self.n_splits = config['training']['n_splits']
    
    def get_cv_split(self, split_path):
        cv = KFoldCrossTrial(n_splits=self.n_splits, shuffle=True, split_path=split_path)
        for train_dataset, eval_dataset in cv.split(self.dataset):
            # The total number of experiments is the number subjects multiplied by K
            train_loader = self.make_loader(dataset=train_dataset, batch_size=self.train_batch_size)
            eval_loader = self.make_loader(dataset=eval_dataset, batch_size=self.eval_batch_size)

            yield eval_loader, train_loader
        
    def make_loader(self, batch_size, dataset=None):
        if dataset is None:
            dataset = self.dataset
        return torch.utils.data.DataLoader(
                            dataset,
                            batch_size=batch_size,
                            num_workers=self.num_worker,
                            pin_memory=True,
                            drop_last=True,
                            persistent_workers=False
                        )
