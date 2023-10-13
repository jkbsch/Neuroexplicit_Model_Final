import json
import argparse
import warnings

from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader




class Trans_Matr_Static():
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold

        self.cfg = config
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']

        
        self.train_iter = 0
        self.labels = self.build_dataloader()

        self.train_labels = self.labels['train']
        self.val_labels = self.labels['val']

        self.trans_matr = self.build_trans_matr()



    
    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        #print('train_dataset.labels: ', train_dataset.labels)

        return {'train': train_dataset.labels, 'val': val_dataset.labels}

    def build_trans_matr(self):
        trans_matr = np.zeros((5,5))
        for i in range(len(self.train_labels)):
            for j in range(len(self.train_labels[i])-1):
                a = self.train_labels[i][j]
                b = self.train_labels[i][j+1]
                trans_matr[a][b] += 1

        sum_per_ax = np.sum(trans_matr, axis=1)

        for i in range(len(sum_per_ax)):
            break

        print(sum_per_ax) ## missing calculation of probabilities




        print(trans_matr)
        print("Hey!")






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
    

    trans_matrix = Trans_Matr_Static(args, 1, config)


    #trans_matrix = calc_trans_matr_stat(train_labels, val_labels)
    print(trans_matrix.train_labels)
    print("Hi")


if __name__ == "__main__":
    main()
