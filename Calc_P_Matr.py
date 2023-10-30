import json
import argparse
import warnings
from copy import deepcopy

import numpy as np
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
        self.lengths = []
        self.set = 'train'
        
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
        self.set = 'test'
        P_dataset = EEGDataLoader(self.cfg, self.fold, set=self.set)  # muss wieder zu train werden - hier möchte ich ja nicht mit dem Test-Dataset trainieren, was ich später noch zum evaulieren brauche
        """P_dataset_zero = deepcopy(P_dataset)
        P_dataset_zero.labels = P_dataset_zero.labels[0]
        P_dataset_zero.labels = P_dataset_zero.labels[np.newaxis, :]
        P_dataset_zero.inputs = P_dataset_zero.inputs[0]
        P_dataset_zero.inputs = P_dataset_zero.inputs[np.newaxis, :]
        P_dataset_zero.epochs = P_dataset_zero.epochs[0:len(P_dataset_zero.labels[0])]"""

        for i in range(len(P_dataset.labels)):
            self.lengths.append(len(P_dataset.labels[i]))

        P_loader = DataLoader(dataset=P_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')

        return {'P': P_loader} # hier könnte man noch die Länge des Datasets hinzufügen und mit floor berechnen, aber das ergäbe für [0] 16, also müsste es doch für 15 noch laufen?
        #return self.build_dataloader_separate()

    """def build_dataloader_separate(self):  # Problem von mir: ich trainiere SleePyCo auf Daten und generiere dann meine P-Matrix
        # mit diesem trainierten Modell auf den Trainingsdaten, nicht auf Test-Daten oder so - Gefahr des Overfitting
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
                              num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True, drop_last=True)
        print('[INFO] Dataloader prepared')

        return {'P': P_loader} """
   
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

    for fold in range(1, config['dataset']['num_splits'] + 1):
    
        Y_true = np.zeros(0)
        Y_pred = np.zeros((0, config['classifier']['num_classes']))
        Y_probs = np.zeros((0, config['classifier']['num_classes']))

        evaluator = OneFoldEvaluator(args, fold, config)
        y_true, y_pred, y_probs = evaluator.run()
        Y_true = np.concatenate([Y_true, y_true])  # wenn i auf 14 gesetzt: hier 960 Elemente
        Y_pred = np.concatenate([Y_pred, y_pred])
        Y_probs = np.concatenate([Y_probs, y_probs])

        prev = 0
        for i in range(len(evaluator.lengths)):
            out_name = "./Probability_Data/"+"_set_"+evaluator.set+"_fold_"+str(evaluator.fold)
            current = prev + evaluator.lengths[i]
            np.savetxt(out_name+"_nr_"+str(i)+"_labels.txt", (Y_true[prev:current]), fmt="%d", delimiter=",")
            np.savetxt(out_name+"_nr_"+str(i)+"_pred.txt", (Y_pred[prev:current]), fmt="%.15f", delimiter=",")
            np.savetxt(out_name+"_nr_"+str(i)+"_probs.txt", (Y_probs[prev:current]), fmt="%.15f", delimiter=",")
            prev = evaluator.lengths[i]

    print("Hey!")
    ## Fold verstanden: wenn fold = 1, dann werden die Daten fürs Testen genommen, die fürs Trainieren von Fold
    # 1 ausgelassen wurden; außerdem wird dann der Checkpoint genutzt, der erstellt wurde durch das Trainieren von Fold1 ? Nicht ganz sicher, warum gibt's dann Checkpoints aber nur eine 10-fold CV?


if __name__ == "__main__":
    main()
