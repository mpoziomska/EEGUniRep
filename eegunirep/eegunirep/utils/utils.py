import os
import toml
import pathlib
import logging
import torch
import numpy as np


def load_config(pth):
    if os.path.exists(pth):
        config_path = pth
    else:
        this_dir = os.path.dirname(__file__)
        config_path = pathlib.Path(os.path.join(this_dir, '..', "configs", f"{pth}.toml"))
    config = toml.load(str(config_path))
    return config

def overwrite_prompt(folder):
    folder = os.path.abspath(folder)
    answer = ''
    while answer not in ['y', 'n']:
        print(f"Are you sure you want to delete {folder}?\n"
              f"y/n")
        try:
            answer = input()
        except Exception:
            continue
        if answer == 'y':
            return True
        else:
            return False

def get_logger(name, loglevel):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        consoleHandler = logging.StreamHandler()
        logFormatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
        logger.setLevel(loglevel)
    return logger

def get_debug_info(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.copy(x))
    return f"{x.shape}, nans: {torch.isnan(x).sum()}, max: {x.max()}, min: {x.min()}"
