import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from HMM_utils import *
from Viterbi.Viterbi_Algorithm import *


class OptimTransMatrix:
    def __init__(self, dataset='Sleep-EDF-2013', checkpoints='given', trans_matrix='EDF_2013', fold=1, num_epochs=2,
                 learning_rate=0.0000001, alpha=None, train_transition=True, train_alpha=False, save=False,
                 print_info=True, print_results=False, save_unsuccesful=False, use_normalized=True):

        # Device configuration
        # torch.autograd.set_detect_anomaly(True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.TrainDataset = self.TrainSleepDataset(self.device, dataset, checkpoints, trans_matrix, fold)
        self.dataset = self.TrainDataset.dataset
        self.trans_matrix = self.TrainDataset.trans_matrix
        self.fold = self.TrainDataset.fold
        self.checkpoints = self.TrainDataset.checkpoints
        self.train_alpha = train_alpha
        self.alpha = alpha
        self.save_alpha = alpha
        self.train_transition = train_transition
        self.print_info = print_info
        self.print_results = print_results
        self.no_nan = True
        self.save_unsuccesful = save_unsuccesful
        self.use_normalized = use_normalized

        self.TestDataset = self.TestSleepDataset(self.device, self.dataset, self.checkpoints, self.trans_matrix,
                                                 self.fold)
        self.train_loader = DataLoader(dataset=self.TrainDataset, batch_size=1, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(dataset=self.TestDataset, batch_size=1, shuffle=True, num_workers=2)
        # self.one_hot = nn.functional.one_hot(self.labels)
        self.trans = self.trainable()

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.loss = nn.CrossEntropyLoss()

        train_params = []
        if train_transition:
            train_params.append(self.trans)
        if train_alpha:
            train_params.append(self.alpha)
        self.optimizer = torch.optim.Adam(train_params, lr=self.learning_rate)

        self.successful = self.training()

        if save:
            self.save(save_unsuccesful)

    class TrainSleepDataset(Dataset):
        def __init__(self, device, dataset='Sleep-EDF-2013', checkpoints='given', trans_matrix='EDF_2013', fold=1):
            self.checkpoints = checkpoints
            self.used_set = 'train'
            self.fold = fold
            self.device = device
            self.dataset, self.trans_matrix, self.end_fold, self.end_nr, _ = set_dataset('train',
                                                                                         dataset, trans_matrix)
            self.end_nr += 1
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
            self.dataset, self.trans_matrix, self.end_fold, self.end_nr, self.leave_out = set_dataset('test',
                                                                                                      dataset,
                                                                                                      trans_matrix)

            while (self.fold, self.end_nr) in self.leave_out:
                self.end_nr -= 1
            self.end_nr += 1
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

    def trainable(self):
        trans = torch.from_numpy(load_Transition_Matrix(self.trans_matrix)).to(device=self.device, dtype=torch.float64)
        if self.train_transition:
            trans.requires_grad_()

        if self.train_alpha:
            if self.alpha is None:
                self.alpha = 0.5
            self.alpha = torch.tensor([self.alpha], requires_grad=True, dtype=torch.float64, device=self.device)

        return trans

    def forward(self, data):
        res = Viterbi(self.trans, data, alpha=self.alpha, logscale=True, return_log=True, print_info=False)
        if self.use_normalized:
            res_normalized = torch.div(res.T1, torch.sum(res.T1, dim=0))
        else:
            res_normalized = None
        return res.x, res.T1, res_normalized

    def train(self, epoch):
        nr, total_loss, total_acc = 0, 0, 0
        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs = torch.squeeze(inputs, dim=0)
            targets = torch.squeeze(targets, dim=0)
            labels_predicted, y_predicted_unnormalized, y_predicted_normalized = self.forward(inputs)

            """if i % 10 == 0:
                print(self.trans)"""

            if self.use_normalized:
                pred = y_predicted_normalized
            else:
                pred = y_predicted_unnormalized
            loss = self.loss(torch.transpose(pred, 0, 1), targets)

            if torch.isnan(loss):
                if self.print_info:
                    print("[INFO]: loss is NaN, training was stopped. Epoch:" + str(epoch))
                return False

            # calculate gradients = backward pass
            loss.backward()

            # update weights
            self.optimizer.step()

            # zero the gradients after updating
            self.optimizer.zero_grad()
            if epoch % 10 == 0 and self.print_results:
                total_acc += (labels_predicted == targets).sum().item() / len(labels_predicted)
                total_loss += loss.item()
                nr += 1
                if i == self.TrainDataset.end_nr - 1:
                    total_acc /= nr
                    total_loss /= nr
                    # print(f"i {i + 1} new Trans Matrix = {self.trans} Training loss: {loss.item():>7f}")
                    if self.print_results:
                        if torch.is_tensor(self.alpha):
                            alpha = self.alpha.item()
                        else:
                            alpha = self.alpha
                        print(f"Epoch: {epoch}, i = {i + 1}, Alpha = {alpha:.5f}  \nTrain Accuracy (Average): "
                              f"\t{(100 * total_acc):0.5f}%\tTrain loss (Average): \t{total_loss:.7f}")

        return True

    def test(self, epoch):
        test_loss, correct = 0, 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                inputs = torch.squeeze(inputs, dim=0)
                targets = torch.squeeze(targets, dim=0)
                labels_predicted, y_predicted_unnormalized, y_predicted_normalized = self.forward(inputs)

                if self.use_normalized:
                    test_loss += self.loss(torch.transpose(y_predicted_normalized, 0, 1), targets).item()
                else:
                    test_loss += self.loss(torch.transpose(y_predicted_unnormalized, 0, 1), targets).item()
                correct += (labels_predicted == targets).sum().item() / len(labels_predicted)

            test_loss /= self.TestDataset.end_nr
            correct /= self.TestDataset.end_nr

            if epoch % 10 == 0:
                if self.print_results:
                    print(f"Test Accuracy (Average): \t{(100 * correct):0.5f}%\tTest loss (Average): \t{test_loss:.6f}")

    def training(self):
        if self.print_info:
            print("[INFO]: Data has been loaded. Training starts.")
        for epoch in range(self.num_epochs):
            successful = self.train(epoch)
            if not successful:
                if self.save_unsuccesful:
                    self.no_nan = False
                    continue
                else:
                    return False
            self.test(epoch)

        return True

    def save(self, save_unsuccessful):
        if self.successful and self.no_nan and not save_unsuccessful:
            out_name = ("./Transition_Matrix/optimized_" + self.dataset + "_fold_" + str(self.fold) + "_checkpoints_" +
                        str(self.checkpoints) +"_lr_"+str(self.learning_rate) + "_alpha_"+str(self.save_alpha)+"_epochs_"+str(self.num_epochs) + ".txt")
            np.savetxt(out_name, self.trans.detach().numpy(), fmt="%.15f", delimiter=",")
            if self.print_info:
                print("[INFO]: Training of fold " + str(self.fold) + " of the dataset '" + self.dataset +
                      "' was successful. Data has been saved")
        else:
            if save_unsuccessful:
                out_name = ("./Transition_Matrix/optimized_" + self.dataset + "_fold_" + str(self.fold) +
                            "_checkpoints_" + str(self.checkpoints)+"_lr_"+str(self.learning_rate) + "_alpha_"+str(self.save_alpha)+"_epochs_"+str(self.num_epochs) + "_unsuccessful.txt")
                np.savetxt(out_name, self.trans.detach().numpy(), fmt="%.15f", delimiter=",")
                if self.print_info:
                    print("[INFO]: training was not completed. Transition matrix has not passed all epochs.")
            else:
                if self.print_info:
                    print("[INFO]: training was not successful. Transition matrix will not be saved.")


def main():
    for fold in range(1, 21):
        OptimTransMatrix(dataset='Sleep-EDF-2013', num_epochs=60, learning_rate=0.0001, print_results=True,
                         train_alpha=False, train_transition=True, alpha=0.3, fold=fold, save=False,
                         save_unsuccesful=False, use_normalized=True)
    for fold in range(1, 11):
        OptimTransMatrix(dataset='Sleep-EDF-2018', num_epochs=60,
                        learning_rate=0.0001, print_results=True, train_alpha=False, train_transition=True, alpha=0.3, fold=fold,
                        save=True, use_normalized=True, save_unsuccesful=False)


if __name__ == "__main__":
    main()
