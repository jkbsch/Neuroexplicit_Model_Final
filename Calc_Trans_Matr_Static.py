import json
import os
import numpy as np
from loader import EEGDataLoader


class TransMatrStatic:
    """
    TransMatrStatic: class; instances contain the transition matrix derived from the chosen datasets;
    Transition matrices are solely based on the statistical averages over the datasets
        """
    def __init__(self, edf_2013=True, edf_2018=True, mass=False, shhs=False, physio_2018=False):

        self.edf_2013 = edf_2013  # the datasets that should be used for calculating the trans matrix can be chosen
        self.edf_2018 = edf_2018
        self.mass = mass
        self.shhs = shhs
        self.physio_2018 = physio_2018

        self.nr_datasets = 0  # number of datasets selected (is set in self.build_dataset_configs())

        self.dataset_configs = self.build_dataset_configs()  # configuration data used for loading data is set

        self.total_labels = self.build_total_labels()  # list of dicts containing the name of the dataset and its labels

        self.all_trans_matr = self.build_trans_matr()  # separate transition matrix for every dataset

        self.average_trans_matr = self.build_average_trans_matr()  # average over all transition matrices selected
        print("Hey!")

    def build_dataset_configs(self):
        """"
        Function creates and returns a list of dictionaries containing the name of the dataset and its according
        configuration files required for loading the dataset
        """
        dataset_args = []  # list is initiated and arguments are appended
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

        self.nr_datasets = len(dataset_args)  # nr_datasets is set
        assert (self.nr_datasets > 0)  # assert that at least one dataset is selected

        dataset_configs = []  # initialize list of dicts for configs_dataset

        for i in range(len(dataset_args)):  # config data from the config file is added to the list of dicts
            with open(dataset_args[i]["config"]) as config_file:
                config = json.load(config_file)
            config['name'] = os.path.basename(dataset_args[i]["config"]).replace('.json', '')
            dataset_configs.append({"dataset": dataset_args[i]["dataset"], "config": config})

        return dataset_configs

    def build_dataloader(self, i):
        """
        Labels from the i-th dataset in the dataset_configs list are returned, separated by train and validation
        """
        train_dataset = EEGDataLoader(self.dataset_configs[i]["config"], 1, set='train')
        val_dataset = EEGDataLoader(self.dataset_configs[i]["config"], 1, set='val')

        return {'train': train_dataset.labels, 'val': val_dataset.labels}

    def build_total_labels(self):
        """
        Creates a list of dicts containing the name of the dataset and its labels;
        """

        total_labels = []  # list is initiated
        for i in range(len(self.dataset_configs)):  # for every dataset selected
            labels = self.build_dataloader(i)  # load data
            train_labels = labels["train"]
            val_labels = labels["val"]
            train_labels.extend(val_labels)  # combine train and val data (separation not required for trans matrix)

            total_labels.append({"dataset": self.dataset_configs[i]["dataset"], "labels": train_labels})

        return total_labels

    def build_trans_matr(self):
        """
        Builds separate transition matrices for every dataset and stores them in a list of dicts;
        """
        all_trans_matr = []  # list initiated

        for k in range(self.nr_datasets):  # for every dataset selected
            trans_matr = np.zeros((5, 5), dtype=int)  # create empty transition matrix

            current_labels = self.total_labels[k]["labels"]
            dataset = self.total_labels[k]["dataset"]

            for i in range(len(current_labels)):  # loop over all subjects in the dataset

                for j in range(len(current_labels[i]) - 1):  # loop over every epoch within a subject's data
                    a = current_labels[i][j]  # current sleep stage
                    b = current_labels[i][j + 1]  # next sleep stage
                    trans_matr[a][b] += 1  # increase counter of transitions in the transition matrix

            nr_epochs = np.sum(trans_matr)  # calculate the total number of epochs (not exact) in this dataset (
            # required for weighing the average matrix all datasets);
            row_sums = trans_matr.sum(axis=1)  # adapt transition matrix to show probabilities
            normalized_trans_matr = trans_matr / row_sums[:, np.newaxis]

            assert np.all((np.round(normalized_trans_matr.sum(axis=1), decimals=3) == 1.0))  # trans matrices represent
            # transition probabilities and thus each row should add up to one

            all_trans_matr.append({"dataset": dataset, "trans_matr": normalized_trans_matr,
                                   "nr_epochs": nr_epochs})

        return all_trans_matr

    def build_average_trans_matr(self):
        """
        Returns the average transition matrix over all transition matrices calculated.
        Every transition matrix is weighted with its number of epochs
        """

        total_epochs = 0  # total number of epochs in the dataset
        average_trans_matr = np.zeros((5, 5))  # init empty total transition matrix
        assert (self.nr_datasets == len(self.all_trans_matr))  # assert that the number of datasets stored == number of
        # datasets in the all_trans_matr list
        for i in range(self.nr_datasets):  # count the total number of epochs over all datasets
            total_epochs += self.all_trans_matr[i]["nr_epochs"]

        for i in range(self.nr_datasets):  # all transition matrices are added and weighted
            average_trans_matr += self.all_trans_matr[i]["trans_matr"]*(self.all_trans_matr[i]["nr_epochs"]/total_epochs)

        assert np.all((np.round(average_trans_matr.sum(axis=1), decimals=3) == 1.0))  # trans matrices represent
        # transition probabilities and thus each row should add up to one
        return average_trans_matr




def main():
    trans_matrix = TransMatrStatic(edf_2013=True, edf_2018=True, mass=False, shhs=False, physio_2018=False)
    #out_name = "./Transition_Matrix/"
    #np.savetxt(out_name + "edf-2013-and-edf-2018.txt",trans_matrix.average_trans_matr, fmt="%.15f", delimiter=",")


if __name__ == "__main__":
    main()
