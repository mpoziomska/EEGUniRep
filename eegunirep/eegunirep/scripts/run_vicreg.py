import argparse
from shutil import rmtree
import json
import torch

from eegunirep.loaders import VicRegLoader
from eegunirep.utils.utils import load_config, overwrite_prompt
from eegunirep.training import VicRegTrain


def get_arguments():
    parser = argparse.ArgumentParser(description="Train a custom model with VICReg")

    # Data
    parser.add_argument("-d", "--data-dir", type=str, default="/home/martynapoziomska/cached_eeg/final",
                        help='Path to io cached folder for torcheeg')
    
    parser.add_argument("-csv", "--csv-pth", type=str, default=None,
                        help='Path to csv info')
    
    parser.add_argument("-oc", "--overwrite-cached", default=False, action='store_true',
                        help='Overwrite cached data with new settings')
    
    parser.add_argument("-c", "--config-pth", type=str, default="base",
                        help='Path to config')
    
    parser.add_argument("-st", "--skip-training", default=False, action='store_true',
                        help='Skip training, do only preprocessing')
    
    parser.add_argument("--deb", default=False, action='store_true',
                        help='Debug level')
    
    parser.add_argument("--profiler", default=False, action='store_true',
                        help='Run profiler')
    
    return parser.parse_args()
    
def main():
    args = get_arguments()
    config = load_config(args.config_pth)
    print(json.dumps(config, indent=4, ensure_ascii=False))
    print(args)


    
    if args.overwrite_cached:
        remove = overwrite_prompt(args.data_dir)
        if remove:
            print(f'Deleting {args.data_dir}')
            rmtree(args.data_dir)
            print('Done')
    loglevel = 'DEBUG' if args.deb else 'INFO'
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype=torch.float32
    
    loader = VicRegLoader(csv_path=args.csv_pth, config=config, io_path=args.data_dir, device=device, dtype=dtype, profiler=args.profiler, loglevel=loglevel)

    if not args.skip_training:
        training_engine = VicRegTrain(loader=loader, config=config, device=device, dtype=dtype, profiler=args.profiler, loglevel=loglevel)
        training_engine.make_CV()


if __name__ == '__main__':
    main()