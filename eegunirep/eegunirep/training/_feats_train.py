import glob
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from eegunirep.utils.utils import get_logger, get_debug_info, overwrite_prompt
from eegunirep.loaders import RawLoader
from eegunirep.models import FeatureClassification
from sklearn.model_selection import GroupKFold

class FeatsTrain:
    
    def __init__(self, config, device, loglevel, fold_pth, data_dir, profiler=False, dtype=torch.float32):
        self.profiler = profiler
        self.loglevel = loglevel
        self.logger = get_logger(name="FEAT TRAIN", loglevel=loglevel)
        self.device = device
        self.device_name = 'cuda' if self.device == torch.device('cuda:0') else 'cpu'
        self.dtype = dtype
        self.config = config
        self.fold_pth = Path(fold_pth)
        self.data_dir = data_dir
        self.label_name = config['feature_classification']['cls']
        self.n_splits = config['feature_classification']['n_splits']
        self.loader = RawLoader(csv_path=fold_pth, config=config, io_path=data_dir, device=device, dtype=dtype, profiler=profiler, loglevel=loglevel).loader

    def get_feats(self, save, model, epoch):
        if model is None:
            fold_nr = self.fold_pth.stem.split('_')[-1]
            model_pth = self.fold_pth.parent.parent / f'models/best_fold_{fold_nr}.pt'
            model = torch.load(model_pth).to(device=self.device, dtype=self.dtype)
        
        with torch.no_grad():
            model.eval()
            representations = []
            embeddings = []
            info_list = None
            for x, label, info in self.loader:
                # print(f"X: {x.shape}")
                with torch.amp.autocast(self.device_name):
                    rep_x, emb_x = model(x=x.to(device=self.device, dtype=self.dtype), batch_size=self.config['training']['eval_batch_size'])
                representations += [rep_x]
                embeddings += [emb_x]
                self.logger.debug(f"X: {x.shape}, rep: {rep_x.shape}, emb: {emb_x.shape}")

                if info_list is not None:
                    for key, value in info.items():
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().detach().numpy().tolist()
                        info_list[key] += value
                else:
                    info_list = {}
                    for key, value in info.items():
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().detach().numpy().tolist()

                        
                        info_list[key] = value
                
            representations = torch.cat(representations).detach().cpu().numpy()
            embeddings = torch.cat(embeddings).detach().cpu().numpy()

            info = pd.DataFrame(info_list)
            Y = info[self.label_name].values
            if save:
                fold_nr = self.fold_pth.stem.split('_')[-1]
                save_pth = self.fold_pth.parent.parent / f'emb_rep/fold_{fold_nr}_epoch_{epoch}'
                save_pth.mkdir(exist_ok=True, parents=True)

                np.save(save_pth / "repr.npy", representations)
                
                np.save(save_pth / "emb.npy", embeddings)

                info.to_csv(save_pth / "info.csv")

            return representations, Y, info

    def make_CV(self, save=False, vicreg_model=None, epoch='best'):
        X, Y, info = self.get_feats(save=save, model=vicreg_model, epoch=epoch)
        self.logger.debug(f"X: {X.shape}, y: {Y.shape}, info: {len(info)}")
        group_kfold = GroupKFold(n_splits=self.n_splits)

        score_list = []
        for train_index, test_index in tqdm(group_kfold.split(X, Y, info['institution_id']), desc='Training feat model...'):
            X_train = X[train_index]
            y_train = Y[train_index]
            X_test = X[test_index]
            y_test = Y[test_index]

            model = FeatureClassification(config=self.config, loglevel=self.loglevel)

            score = model.train_eval(X_train, y_train, X_test, y_test)

            score_list += [score]

        return np.mean(score_list), np.std(score_list)

        