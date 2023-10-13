
import json
import argparse
import warnings

from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']

        
        self.train_iter = 0
        self.loader_dict = self.build_dataloader()



    
    def build_dataloader(self):
        dataloader_args = {'batch_size': self.tp_cfg['batch_size'], 'shuffle': True, 'num_workers': 4*len(self.args.gpu.split(",")), 'pin_memory': True}
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, **dataloader_args)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, **dataloader_args)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader}

    


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    
    for fold in range(1, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)


if __name__ == "__main__":
    main()
