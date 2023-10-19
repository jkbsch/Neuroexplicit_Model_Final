import json
import os
import numpy as np
from loader import EEGDataLoader


class TransMatrStatic:
    def __init__(self, edf_2013=True, edf_2018=True, mass=False, shhs=False, physio_2018=False):

        self.edf_2013 = edf_2013
        self.edf_2018 = edf_2018
        self.mass = mass
        self.shhs = shhs
        self.physio_2018 = physio_2018

        self.nr_datasets = 0

        self.dataset_configs = self.build_dataset_configs()

        self.total_labels = self.build_total_labels()

        self.all_trans_matr = self.build_trans_matr()

        self.average_trans_matr = self.build_average_trans_matr()

    def build_dataset_configs(self):
        dataset_args = []
        if self.edf_2013:
            dataset_args.append({"dataset": "EDF_2013",
                                 "config": "configs/SleePyCo-Transformer_SL-01_numScales-1_Sleep-EDF-2013_pretrain.json"})

        if self.edf_2013:
            dataset_args.append({"dataset": "EDF_2018",
                                 "config": "configs/SleePyCo-Transformer_SL-01_numScales-1_Sleep-EDF-2018_pretrain.json"})

        if self.mass:
            dataset_args.append(
                {"dataset": "MASS", "config": "configs/SleePyCo-Transformer_SL-01_numScales-1_MASS_pretrain.json"})

        if self.shhs:
            dataset_args.append(
                {"dataset": "SHHS", "config": "configs/SleePyCo-Transformer_SL-01_numScales-1_SHHS_pretrain.json"})

        if self.physio_2018:
            dataset_args.append({"dataset": "Physio2018",
                                 "config": "configs/SleePyCo-Transformer_SL-01_numScales-1_Physio2018_pretrain.json"})

        self.nr_datasets = len(dataset_args)
        assert (self.nr_datasets > 0)

        dataset_configs = []

        for i in range(len(dataset_args)):
            with open(dataset_args[i]["config"]) as config_file:
                config = json.load(config_file)
            config['name'] = os.path.basename(dataset_args[i]["config"]).replace('.json', '')
            dataset_configs.append({"dataset": dataset_args[i]["dataset"], "config": config})

        return dataset_configs

    def build_dataloader(self, i):
        train_dataset = EEGDataLoader(self.dataset_configs[i]["config"], 1, set='train')
        val_dataset = EEGDataLoader(self.dataset_configs[i]["config"], 1, set='val')

        return {'train': train_dataset.labels, 'val': val_dataset.labels}

    def build_total_labels(self):

        total_labels = []
        for i in range(len(self.dataset_configs)):
            labels = self.build_dataloader(i)
            train_labels = labels["train"]
            val_labels = labels["val"]
            train_labels.extend(val_labels)

            total_labels.append({"dataset": self.dataset_configs[i]["dataset"], "labels": train_labels})

        return total_labels

    def build_trans_matr(self):
        all_trans_matr = []

        for k in range(self.nr_datasets):
            trans_matr = np.zeros((5, 5), dtype=int)
            nr_epochs = 0

            current_labels = self.total_labels[k]["labels"]
            dataset = self.total_labels[k]["dataset"]

            for i in range(len(current_labels)):
                nr_epochs += len(current_labels[i])

                for j in range(len(current_labels[i]) - 1):
                    a = current_labels[i][j]
                    b = current_labels[i][j + 1]
                    trans_matr[a][b] += 1

            row_sums = trans_matr.sum(axis=1)
            normalized_trans_matr = trans_matr / row_sums[:, np.newaxis]

            all_trans_matr.append({"dataset": dataset, "trans_matr": normalized_trans_matr,
                                   "nr_epochs": nr_epochs})

        return all_trans_matr

    def build_average_trans_matr(self):

        total_epochs = 0
        average_trans_matr = np.zeros((5, 5))
        assert (self.nr_datasets == len(self.all_trans_matr))
        for i in range(self.nr_datasets):
            total_epochs += self.all_trans_matr[i]["nr_epochs"]

        for i in range(self.nr_datasets):
            average_trans_matr += self.all_trans_matr[i]["trans_matr"]*(self.all_trans_matr[i]["nr_epochs"]/total_epochs)

        assert np.all((np.round(average_trans_matr.sum(axis=1), decimals=3) == 1.0))
        return average_trans_matr


def main():
    trans_matrix = TransMatrStatic()
    print("successful")


if __name__ == "__main__":
    main()
