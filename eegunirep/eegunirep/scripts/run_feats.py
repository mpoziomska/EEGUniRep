import argparse

import torch
import glob

from eegunirep.loaders import VicRegLoader
from eegunirep.utils.utils import load_config, overwrite_prompt
from eegunirep.training import FeatsTrain

def get_arguments():
    parser = argparse.ArgumentParser(description="Train a custom model with VICReg")

    parser.add_argument("-d", "--data-dir", type=str, default="/home/martynapoziomska/cached_eeg/final",
                        help='Path to io cached folder for torcheeg')
    
    parser.add_argument("-f", "--fold-pth", type=str, default=None,
                        help='Path to csv info')
    
    parser.add_argument("-c", "--config-pth", type=str, default="base",
                        help='Path to config')
    
    parser.add_argument("--deb", default=False, action='store_true',
                        help='Debug level')
    
    parser.add_argument("--profiler", default=False, action='store_true',
                        help='Run profiler')
    
    return parser.parse_args()
    
def main():
    args = get_arguments()
    config = load_config(args.config_pth)
    
    loglevel = 'DEBUG' if args.deb else 'INFO'
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype=torch.float32
    
    training_engine = FeatsTrain(config=config, fold_pth=args.fold_pth, data_dir=args.data_dir, device=device, dtype=dtype, profiler=args.profiler, loglevel=loglevel)
    training_engine.make_CV()

    
if __name__ == '__main__':
    main()