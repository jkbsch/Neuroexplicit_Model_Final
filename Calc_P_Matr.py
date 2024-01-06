import json
import argparse
import warnings

import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from train_mtcl import OneFoldTrainer
from models.main_model import MainModel


class OneFoldEvaluator(OneFoldTrainer):
    def __init__(self, args, fold, config):  # initialize
        self.args = args
        self.fold = fold

        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        self.lengths = []
        self.set = 'val'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        self.criterion = nn.CrossEntropyLoss()
        self.ckpt_path = os.path.join('checkpoints_copied', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)

    def build_model(self):  # build nn model
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model

    def build_dataloader(self):
        P_dataset = EEGDataLoader(self.cfg, self.fold, set=self.set)
        print("P_dataset labels:", P_dataset.labels)
        for i in range(len(P_dataset.labels)):
            self.lengths.append(len(P_dataset.labels[i]))  # save how many epochs every night has to separate later


        P_loader = DataLoader(dataset=P_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                              num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True)  # create torch dataloader
        print('[INFO] Dataloader prepared')

        return {'P': P_loader}  # return dataloader object as library


    def run(self):
        print('\n[INFO] Fold: {}'.format(self.fold))
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name), map_location=self.device))  # load checkpoints
        y_true, y_pred, y_probs = self.Evalute_P_Matr()  # open Evaluate_P_Matr function in train_mtcl
        print('')
        print("y_true: ", y_true, "y_pred: ", y_pred, "y_probs: ", y_probs)

        return y_true, y_pred, y_probs


def main():
    # read input
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path')
    args = parser.parse_args()


    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Source for setting gpu_devices: ChatGPT -
    # evaluate IDs for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Get the allocated GPU device IDs from the SLURM environment
    gpu_devices = os.environ.get('SLURM_JOB_GPUS', '0')

    # Set CUDA_VISIBLE_DEVICES to the allocated GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    args.gpu = gpu_devices
    print(gpu_devices)
    args.gpu = '0'

    with open(args.config) as config_file:  # load config data
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')

    for fold in range(1, config['dataset']['num_splits'] + 1):  # iterate through all folds

        # initialize empty arrays
        Y_true = np.zeros(0)
        Y_pred = np.zeros((0, config['classifier']['num_classes']))
        Y_probs = np.zeros((0, config['classifier']['num_classes']))

        evaluator = OneFoldEvaluator(args, fold, config)  # create evaluator object
        y_true, y_pred, y_probs = evaluator.run()  # run the evaluator to calculate the probabilities
        Y_true = np.concatenate([Y_true, y_true])  # concatenate all results
        Y_pred = np.concatenate([Y_pred, y_pred])
        Y_probs = np.concatenate([Y_probs, y_probs])
        print("Here 1")
        print(len(evaluator.lengths), evaluator.lengths)

        prev = 0
        for i in range(len(evaluator.lengths)):  # save the results separated by the subject
            out_name = "./Probability_Data/Sleep-EDF-2018-val/" + "_dataset_" + config['dataset'][
                'name'] + "_set_" + evaluator.set + "_fold_" + str(evaluator.fold)
            current = prev + evaluator.lengths[i]
            np.savetxt(out_name + "_nr_" + str(i) + "_labels.txt", (Y_true[prev:current]), fmt="%d", delimiter=",")
            np.savetxt(out_name + "_nr_" + str(i) + "_pred.txt", (Y_pred[prev:current]), fmt="%.15f", delimiter=",")
            np.savetxt(out_name + "_nr_" + str(i) + "_probs.txt", (Y_probs[prev:current]), fmt="%.15f", delimiter=",")

            print(out_name + "_nr_" + str(i) + "_labels.txt", (Y_true[prev:current]))
            print("Here2")
            prev = evaluator.lengths[i]

if __name__ == "__main__":
    main()
