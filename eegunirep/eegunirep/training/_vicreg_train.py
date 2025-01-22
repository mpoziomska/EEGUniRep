from eegunirep.models import BaseVICReg, FeatureClassification
from eegunirep.optimizers import LARS
from eegunirep.utils.utils import get_logger, get_debug_info, overwrite_prompt
from eegunirep.training import FeatsTrain

from tqdm import tqdm
import torch
import math
import neptune
from neptune.utils import stringify_unsupported
import gc
import os
import toml
import pandas as pd
from shutil import rmtree
import numpy as np

from torch.profiler import profile, record_function, ProfilerActivity

class VicRegTrain:

    def __init__(self, loader, config, device, loglevel, profiler=False, dtype=torch.float32):
        self.profiler = profiler
        self.loglevel = loglevel
        self.logger = get_logger(name="TRAIN", loglevel=loglevel)
        self.loader = loader
        self.device = device
        self.device_name = 'cuda' if self.device == torch.device('cuda:0') else 'cpu'
        self.dtype = dtype
        self.config = config
        self.train_batch_size = config['training']['train_batch_size']
        self.eval_batch_size = config['training']['eval_batch_size']
        self.epochs = config['training']['epochs']
        self.base_lr = config['training']['base_lr']
        self.make_feature_classification = config['feature_classification']['make_classification']
        self.epoch_stride = config['feature_classification']['epoch_stride']
        self.exp_pth = f"{config['general']['runs_pth']}/{config['general']['exp_id']}"
        try:
            os.makedirs(self.exp_pth)
        except FileExistsError:
            remove = overwrite_prompt(self.exp_pth)
            if remove:
                print(f'Deleting {self.exp_pth}')
                rmtree(self.exp_pth)
                print('Done')
            os.makedirs(self.exp_pth)
        os.makedirs(f"{self.exp_pth}/models")
        os.makedirs(f"{self.exp_pth}/results")

        with open(f"{self.exp_pth}/config.toml", 'w') as f:
            toml.dump(self.config, f=f)
        
        self.metric_list = ["loss", "repr_loss", "std_loss", "cov_loss", "feat_score"]
        self.neptune_run = neptune.init_run(
            name=config['general']['exp_id'],
            project=config['neptune']['project'],
            api_token=config['neptune']['api_token'],
        )
        self.neptune_run["parameters"] = stringify_unsupported(config)

    def make_CV(self):
        columns = ['epoch'] + [f"train_{x}" for x in self.metric_list] + [f"eval_{x}" for x in self.metric_list]
        fold_best_loss = None # pd.DataFrame(data=None, columns=['fold', 'best_epoch'] + self.metric_list)

        for fold, [eval_loader, train_loader] in enumerate(self.loader.get_cv_split(split_path=f"{self.exp_pth}/splits")):
            self.model = BaseVICReg(device=self.device, dtype=self.dtype, config=self.config, loglevel=self.loglevel)

            self.optimizer = LARS(
                    self.model.parameters(),
                    lr=0,
                    weight_decay_filter=self.exclude_bias_and_norm,
                    lars_adaptation_filter=self.exclude_bias_and_norm,
                )
            self.scaler = torch.cuda.amp.GradScaler()
            best_loss = [1000] * len(self.metric_list)
            best_epoch = 0
            learning_hist = None # pd.DataFrame(data=None, columns=columns)
            self.feats_train = FeatsTrain(config=self.config, device=self.device, loglevel=self.loglevel, fold_pth=f"{self.exp_pth}/splits/test_fold_{fold}.csv", data_dir=self.loader.data_dir, profiler=self.profiler, dtype=self.dtype)
            for epoch in range(1, self.epochs + 1):
                print(f"EPOCH {epoch}/{self.epochs}")
                train_met = self.train_model(train_loader, fold, epoch=epoch)
                eval_met = self.eval_model(eval_loader, fold, epoch=epoch, feature_classification=(self.make_feature_classification and epoch%self.epoch_stride==0))
                new_row = pd.DataFrame(data=[[epoch] + train_met + eval_met], columns=columns)
                learning_hist = pd.concat([learning_hist, new_row], ignore_index=True)
                if eval_met[0] < best_loss[0]:
                    best_loss = eval_met
                    best_epoch = epoch
                    self.logger.info(f"Saving best model ...")
                    torch.save(self.model, f"{self.exp_pth}/models/best_fold_{fold}.pt")
                    self.logger.info(f"Saved")

            del eval_loader, train_loader
            learning_hist.to_csv(f"{self.exp_pth}/results/fold_{fold}_learning_hist.csv")
            new_row = pd.DataFrame(data=[[fold, best_epoch] + best_loss], columns=['fold', 'best_epoch'] + self.metric_list)
            fold_best_loss = pd.concat([fold_best_loss, new_row], ignore_index=True)
            
            gc.collect()
            torch.cuda.empty_cache()
        
        print(fold_best_loss)
        msg = ""
        for met in self.metric_list:
            mean_met = fold_best_loss[met].mean()
            std_met = fold_best_loss[met].std()
            msg += f"{met}: {mean_met:.3f}+={std_met:.3f}"
            self.neptune_run[f'final/{met}/mean'] = mean_met
            self.neptune_run[f'final/{met}/std'] = std_met

        self.neptune_run.stop()

    def train_model(self, loader, fold, epoch):
        len_steps = len(loader)
        self.model.train()
        
        metrics_dict = {x: 0 for x in self.metric_list}
        for step, [x, label, info] in enumerate(tqdm(loader, desc="Train")):
            s1 = x[0].to(device=self.device, dtype=self.dtype, non_blocking=True)
            s2 = x[1].to(device=self.device, dtype=self.dtype, non_blocking=True)
            self.logger.debug(f"s1 {get_debug_info(s1)}") 
            self.logger.debug(f"s2 {get_debug_info(s2)}") 

            # del x
            # gc.collect()
            # self.logger.info(f"mean: {s1.mean(axis=[-1, -2])}, std: {s1.std(axis=[-1, -2])}")
            self.adjust_learning_rate(self.train_batch_size, self.epochs, self.base_lr, self.optimizer, loader, step)
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(self.device_name):
                _, _, emb_x, emb_y = self.model(x=s1, y=s2, batch_size=self.train_batch_size)
                repr_loss, std_loss, cov_loss = self.model.get_loss(emb_x, emb_y, self.train_batch_size)

            loss = repr_loss + std_loss + cov_loss
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            metrics_dict["loss"] += loss.item()
            metrics_dict["repr_loss"] += repr_loss.item()
            metrics_dict["std_loss"] += std_loss.item()
            metrics_dict["cov_loss"] += cov_loss.item()

        return self.update_metrics(metrics_dict=metrics_dict, len_steps=len_steps, fold=fold, epoch=epoch, stage='train')

    def eval_model(self, loader, fold, epoch, feature_classification=False):
        with torch.no_grad():
            self.model.eval()
            len_steps = len(loader)
            metrics_dict = {x: 0 for x in self.metric_list}
            for step, [x, label, info] in enumerate(tqdm(loader, desc="Eval")):
                s1 = x[0].to(device=self.device, dtype=self.dtype, non_blocking=True)
                s2 = x[1].to(device=self.device, dtype=self.dtype, non_blocking=True)
                # del x
                # gc.collect()

                with torch.amp.autocast(self.device_name):
                    rep_x, rep_y, emb_x, emb_y = self.model(x=s1, y=s2, batch_size=self.eval_batch_size)
                    repr_loss, std_loss, cov_loss = self.model.get_loss(emb_x, emb_y, self.eval_batch_size)
                
                loss = repr_loss + std_loss + cov_loss
                
                metrics_dict["loss"] += loss.item()
                metrics_dict["repr_loss"] += repr_loss.item()
                metrics_dict["std_loss"] += std_loss.item()
                metrics_dict["cov_loss"] += cov_loss.item()

            if feature_classification:
                feat_scores = self.feats_train.make_CV(save=True, vicreg_model=self.model, epoch=epoch)
            else:
                feat_scores = [0, 0]
            
            return self.update_metrics(metrics_dict=metrics_dict, len_steps=len_steps, fold=fold, stage='eval', epoch=epoch, feat_scores=feat_scores)
            

    def update_metrics(self, metrics_dict, len_steps, fold, stage, epoch, feat_scores=[0, 0]):
        msg = []
        met_list = []
        for metric, value in metrics_dict.items():
            if metric != 'feat_score':
                value /= len_steps
                self.neptune_run[f"fold_{fold}/{stage}/{metric}"].append(value=value, step=epoch)
                msg += [f"{metric}: {value:.3f}"]
                met_list += [value]
            else:
                model_name = self.config['feature_classification']['model']['name']
                if feat_scores[0] != 0:
                    self.neptune_run[f"fold_{fold}/{stage}/{model_name}/mean"].append(value=feat_scores[0], step=epoch)
                    self.neptune_run[f"fold_{fold}/{stage}/{model_name}/std"].append(value=feat_scores[1], step=epoch)
                met_list += [feat_scores[0]]
                msg += [f"{model_name}: {feat_scores[0]:.3f}+={feat_scores[1]:.3f}"]
            
        msg = ", ".join(msg)
        print(f"{stage}: {msg}")
        return met_list
    
    @staticmethod
    def exclude_bias_and_norm(p):
        return p.ndim == 1

    @staticmethod
    def adjust_learning_rate(batch_size, epochs, base_lr, optimizer, loader, step):
        max_steps = epochs * len(loader)
        warmup_steps = 10 * len(loader)
        base_lr = base_lr * batch_size / 256
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    