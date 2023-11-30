# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from HMM_utils import *
from Viterbi.Viterbi_Algorithm import *


# Linear regression
# f = w * x

# here : f = 2 * x

class FirstOptimTransmatrix:
    def __init__(self, dataset='Sleep-EDF-2013', checkpoints='given', trans_matrix='EDF_2013', fold=1, num_epochs=2,
                 learning_rate=0.0000001):

        # Device configuration
        # torch.autograd.set_detect_anomaly(True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.sleepy, self.labels = self.load_data()
        self.TrainDataset = self.TrainSleepDataset(self.device, dataset, checkpoints, trans_matrix, fold)
        self.dataset = self.TrainDataset.dataset
        self.trans_matrix = self.TrainDataset.trans_matrix
        self.fold = self.TrainDataset.fold
        self.checkpoints = self.TrainDataset.checkpoints

        self.TestDataset = self.TestSleepDataset(self.device, self.dataset, self.checkpoints, self.trans_matrix,
                                                 self.fold)
        self.train_loader = DataLoader(dataset=self.TrainDataset, batch_size=1, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(dataset=self.TestDataset, batch_size=1, shuffle=True, num_workers=2)
        # self.one_hot = nn.functional.one_hot(self.labels)
        self.trans = self.trainable()

        # 2) Define loss and optimizer
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # callable function
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam([self.trans], lr=self.learning_rate)

        self.train()

    class TrainSleepDataset(Dataset):
        def __init__(self, device, dataset='Sleep-EDF-2013', checkpoints='given', trans_matrix='EDF_2013', fold=1):
            self.checkpoints = checkpoints
            self.used_set = 'train'
            self.fold = fold
            self.device = device
            self.dataset, self.trans_matrix, self.end_fold, self.end_nr, _ = set_dataset('train',
                                                                                         dataset, trans_matrix)
            self.train_data_probs = []
            self.train_data_labels = []
            for nr in range(self.end_nr):
                labs, probs = load_P_Matrix(self.checkpoints, self.dataset, self.used_set, self.fold, nr, False)
                labs = torch.from_numpy(labs).to(device=self.device, dtype=torch.int64)
                probs = torch.from_numpy(probs).to(device=self.device, dtype=torch.float64)
                self.train_data_probs.append(probs)
                self.train_data_labels.append(labs)

        def __getitem__(self, idx):
            probs = torch.squeeze(self.train_data_probs[idx], dim=0)
            labs = torch.squeeze(self.train_data_labels[idx], dim=0)
            return probs, labs

        def __len__(self):
            return self.end_nr

    class TestSleepDataset(Dataset):
        def __init__(self, device, dataset='Sleep-EDF-2013', checkpoints='given', trans_matrix='EDF_2013', fold=1):
            self.checkpoints = checkpoints
            self.used_set = 'test'
            self.fold = fold
            self.device = device
            self.dataset, self.trans_matrix, self.end_fold, self.end_nr, _ = set_dataset('test',
                                                                                         dataset, trans_matrix)
            self.test_data_probs = []
            self.test_data_labels = []
            for nr in range(self.end_nr):
                labs, probs = load_P_Matrix(self.checkpoints, self.dataset, self.used_set, self.fold, nr, False)
                labs = torch.from_numpy(labs).to(device=self.device, dtype=torch.int64)
                probs = torch.from_numpy(probs).to(device=self.device, dtype=torch.float64)
                self.test_data_probs.append(probs)
                self.test_data_labels.append(labs)

        def __getitem__(self, idx):
            probs = torch.squeeze(self.test_data_probs[idx], dim=0)
            labs = torch.squeeze(self.test_data_labels[idx], dim=0)
            return probs, labs

        def __len__(self):
            return self.end_nr

    """def load_data(self):
        # 0) Training samples
        P_Matrix_labels, P_Matrix_probs = load_P_Matrix()
        labels = torch.from_numpy(P_Matrix_labels).to(device=self.device, dtype=torch.int64)
        sleepy = torch.from_numpy(P_Matrix_probs).to(device=self.device, dtype=torch.float64)
        return sleepy, labels"""

    def trainable(self):
        # 1) Design Model: Weights to optimize and forward function
        """w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        return w"""
        trans = torch.from_numpy(load_Transition_Matrix(self.trans_matrix)).to(device=self.device, dtype=torch.float64)
        trans.requires_grad_()
        return trans

    def forward(self, data):
        res = Viterbi(self.trans, data, logscale=True, return_log=True, print_info=False)
        res_probs = torch.div(res.T1, torch.sum(res.T1, dim=0))
        return res.x, res.T1, res_probs

    def train(self):
        for epoch in range(self.num_epochs):
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs = torch.squeeze(inputs, dim=0)
                targets = torch.squeeze(targets, dim=0)
                _, y_predicted_unnormalized, y_predicted_normalized = self.forward(inputs)

                loss = self.loss(torch.transpose(y_predicted_unnormalized, 0, 1), targets)

                # calculate gradients = backward pass
                loss.backward()

                # update weights
                self.optimizer.step()

                # zero the gradients after updating
                self.optimizer.zero_grad()

                if i % 10 == 0:
                    print('i ', i + 1, ': new Trans Matrix = ', self.trans, ' loss = ', loss)


def main():
    optimized_trans_matrix = FirstOptimTransmatrix(dataset='Sleep-EDF-2018')
    """out_name = "./Transition_Matrix/"
    np.savetxt(out_name + "first_optimized_trans_matrix.txt", optimized_trans_matrix.trans.detach().numpy(),
               fmt="%.15f", delimiter=",")"""


if __name__ == "__main__":
    main()
