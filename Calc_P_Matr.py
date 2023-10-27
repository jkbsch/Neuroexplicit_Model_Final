import json
import argparse
import warnings
from copy import deepcopy

import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from train_mtcl import OneFoldTrainer
from models.main_model import MainModel


class OneFoldEvaluator(OneFoldTrainer):
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        
        self.criterion = nn.CrossEntropyLoss()
        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        
    def build_model(self):
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        P_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        P_dataset_zero = deepcopy(P_dataset)
        P_dataset_zero.labels = P_dataset_zero.labels[0]
        P_dataset_zero.labels = P_dataset_zero.labels[np.newaxis, :]
        P_dataset_zero.inputs = P_dataset_zero.inputs[0]
        P_dataset_zero.inputs = P_dataset_zero.inputs[np.newaxis, :]
        P_dataset_zero.epochs = P_dataset_zero.epochs[0:len(P_dataset_zero.labels[0])]

        P_loader = DataLoader(dataset=P_dataset_zero, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')

        return {'P': P_loader}
        #return self.build_dataloader_separate()

    def build_dataloader_separate(self):
        P_dataset = EEGDataLoader(self.cfg, self.fold, set='train')

        P_dataset_separate = []
        len_previous = 0

        for i in range(len(P_dataset.labels)):
            P_dataset_separate = P_dataset_separate.extend(deepcopy(P_dataset))
            P_dataset_separate[i].labels = P_dataset_separate[i].labels[i]
            P_dataset_separate[i].labels = P_dataset_separate[i].labels[np.newaxis, :]
            P_dataset_separate[i].inputs = P_dataset_separate[i].inpus[i]
            P_dataset_separate[i].inputs = P_dataset_separate[i].inputs[np.newaxis, :]
            P_dataset_separate[i].epochs = P_dataset_separate[i].epochs[len_previous:len(P_dataset_separate[i].labels[0])]
            len_previous = len(P_dataset_separate[i].labels[0])

        P_loader = DataLoader(dataset=P_dataset_separate[0], batch_size=self.tp_cfg['batch_size'], shuffle=False,
                              num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')

        return {'P': P_loader}
   
    def run(self):
        print('\n[INFO] Fold: {}'.format(self.fold))
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name), map_location=self.device))
        y_true, y_pred, y_probs = self.Evalute_P_Matr()
        print('')

        return y_true, y_pred, y_probs



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

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))
    Y_probs = np.zeros((0, config['classifier']['num_classes']))

    evaluator = OneFoldEvaluator(args, 1, config)
    y_true, y_pred, y_probs = evaluator.run()
    Y_true = np.concatenate([Y_true, y_true])
    Y_pred = np.concatenate([Y_pred, y_pred])
    Y_probs = np.concatenate([Y_probs, y_probs])
    print("Hey!")
    ## Fold verstanden: wenn fold = 1, dann werden die Daten fürs Testen genommen, die fürs Trainieren von Fold
    # 1 ausgelassen wurden; außerdem wird dann der Checkpoint genutzt, der erstellt wurde durch das Trainieren von Fold1


if __name__ == "__main__":
    main()
