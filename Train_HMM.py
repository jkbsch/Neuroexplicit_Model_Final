import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from HMM_utils import *
from Viterbi_Algorithm import *
import argparse
import warnings


class Train_HMM:
    def __init__(self, dataset='Sleep-EDF-2018', checkpoints='given', trans_matrix='EDF_2018', fold=1, num_epochs=2,
                 learning_rate=0.0001, alpha=None, train_transition=True, train_alpha=False, train_with_val=True,
                 save=False,
                 print_info=True, print_results=False, save_unsuccesful=False, use_normalized=False, softmax=False,
                 FMMIE=True, k_best=20, length=1000, min_length=200):

        """
        This class is used to train the transition matrix and the alpha parameter for the Viterbi algorithm
        :param dataset: Name of the dataset
        :param checkpoints: Checkpoint folder used for the DNN, either 'given' or 'own'
        :param trans_matrix: Base Transition matrix used for the Viterbi algorithm
        :param fold: Fold number
        :param num_epochs: Number of epochs used for training
        :param learning_rate: Learning rate used for training
        :param alpha: global parameter Alpha used in the Viterbi algorithm. If alpha is trained, it is the starting value.
                        If alpha is not trained, it is the fixed value.
        :param train_with_val: Whether the validation set is used for training (or the training set)
        :param train_transition: Whether the transition matrix is trained
        :param train_alpha: Whether alpha is trained
        :param save: Whether the transition matrix and alpha are saved
        :param print_info: Whether information is printed
        :param print_results: Whether results are printed
        :param save_unsuccesful: Whether the transition matrix and alpha are saved even if training was not successful because loss is NaN
        :param use_normalized: Whether the normalized version of the Viterbi algorithm is used. Be careful when setting this
                                to True as it is not correctly implemented yet.
        :param softmax: Whether softmax is used in the Viterbi algorithm; Setting this to True might lead to errors with certain parameter settings
        :param FMMIE: Whether the FMMIE loss function is used (or the loss function using the T1 / delta matrix from Viterbi)
        :param k_best: Number of the best paths used for the FMMIE loss function
        :param length: Length of the input data used for the FMMIE loss function
        :param min_length: Minimum length of the input data used for the update step when using the FMMIE loss fucntion (relevant if length < min_length)

        """
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if print_info:
            print(f'Device: {self.device}')

        self.length = length
        self.min_length = min_length
        self.train_with_val = train_with_val

        # create the train dataset
        self.TrainDataset = self.TrainSleepDataset(self.device, dataset, checkpoints, trans_matrix, fold, self.length,
                                                   train_with_val)
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
        self.softmax = softmax
        self.FMMIE = FMMIE
        self.k_best = k_best

        # create the test dataset
        self.TestDataset = self.TestSleepDataset(self.device, self.dataset, self.checkpoints, self.trans_matrix,
                                                 self.fold, self.length, train_with_val)
        # create the dataloaders
        self.train_loader = DataLoader(dataset=self.TrainDataset, batch_size=1, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(dataset=self.TestDataset, batch_size=1, shuffle=True, num_workers=2)

        # set the parameters that should be trained
        self.trans = self.trainable()

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        if self.softmax:
            self.loss = nn.MSELoss()  # this might not be the best loss function for the softmax version
        else:
            self.loss = nn.CrossEntropyLoss()

        train_params = []
        if train_transition:
            train_params.append(self.trans)
        if train_alpha:
            train_params.append(self.alpha)
        self.optimizer = torch.optim.Adam(train_params, lr=self.learning_rate)  # set the optimizer

        self.successful = self.training()  # start training

        if save:
            self.save(save_unsuccesful)  # save the trained transition matrix and alpha

    class TrainSleepDataset(Dataset):
        def __init__(self, device, dataset='Sleep-EDF-2018', checkpoints='given', trans_matrix='EDF_2018', fold=1,
                     length=None, train_with_val=True):
            self.checkpoints = checkpoints
            if train_with_val:
                self.used_set = 'val'
            else:
                self.used_set = 'train'
            self.fold = fold
            self.device = device,
            self.length = length
            self.dataset, self.trans_matrix, self.end_fold, self.end_nr, self.leave_out = set_dataset(self.used_set,
                                                                                                      dataset,
                                                                                                      trans_matrix)
            while (self.fold, self.end_nr) in self.leave_out:
                self.end_nr -= 1
            self.end_nr += 1
            self.train_data_probs = []
            self.train_data_labels = []
            for nr in range(self.end_nr):
                labs, probs = load_P_Matrix(self.checkpoints, self.dataset, self.used_set, self.fold, nr, False)
                labs = torch.from_numpy(labs).to(dtype=torch.int64)
                probs = torch.from_numpy(probs).to(dtype=torch.float64)
                if length is not None and length < len(probs):
                    labs = list(torch.split(labs, length))
                    probs = list(torch.split(probs, length))
                    self.train_data_probs.extend(probs)
                    self.train_data_labels.extend(labs)
                else:
                    self.train_data_probs.append(probs)
                    self.train_data_labels.append(labs)

        def __getitem__(self, idx):
            probs = torch.squeeze(self.train_data_probs[idx], dim=0)
            labs = torch.squeeze(self.train_data_labels[idx], dim=0)
            return probs, labs

        def __len__(self):
            if self.length is None:
                return self.end_nr
            else:
                return len(self.train_data_probs)

    class TestSleepDataset(Dataset):
        def __init__(self, device, dataset='Sleep-EDF-2018', checkpoints='given', trans_matrix='EDF_2018', fold=1,
                     length=10, train_with_val=True):
            self.checkpoints = checkpoints
            if train_with_val:
                self.used_set = 'train'
            else:
                self.used_set = 'val'
            self.fold = fold
            self.device = device,
            self.length = length
            self.dataset, self.trans_matrix, self.end_fold, self.end_nr, self.leave_out = set_dataset(self.used_set,
                                                                                                      dataset,
                                                                                                      trans_matrix)

            while (self.fold, self.end_nr) in self.leave_out:
                self.end_nr -= 1
            self.end_nr += 1
            self.test_data_probs = []
            self.test_data_labels = []
            for nr in range(self.end_nr):
                labs, probs = load_P_Matrix(self.checkpoints, self.dataset, self.used_set, self.fold, nr, False)
                labs = torch.from_numpy(labs).to(dtype=torch.int64)
                probs = torch.from_numpy(probs).to(dtype=torch.float64)
                if length is not None and length < len(probs):
                    labs = list(torch.split(labs, length))
                    probs = list(torch.split(probs, length))
                    self.test_data_probs.extend(probs)
                    self.test_data_labels.extend(labs)
                else:
                    self.test_data_probs.append(probs)
                    self.test_data_labels.append(labs)

        def __getitem__(self, idx):
            probs = torch.squeeze(self.test_data_probs[idx], dim=0)
            labs = torch.squeeze(self.test_data_labels[idx], dim=0)
            return probs, labs

        def __len__(self):
            if self.length is None:
                return self.end_nr
            else:
                return len(self.test_data_probs)

    def trainable(self):
        # load transition matrix and transform it into torch tensor
        trans = torch.from_numpy((load_Transition_Matrix(self.trans_matrix))[1]).to(device=self.device,
                                                                                    dtype=torch.float64)
        if self.train_transition:
            trans.requires_grad_()

        if self.train_alpha:
            if self.alpha is None:
                self.alpha = 1.0  # default alpha = 1
            self.alpha = torch.tensor([self.alpha], requires_grad=True, dtype=torch.float64, device=self.device)

        return trans

    def forward(self, data, targets=None):  # implementation of the forward pass
        trans = torch.clamp(self.trans, min=float(1e-10))
        """in a transition matrix, no entries should be negative. Moreover, entries = 0 might lead to errors due to 
        log(0), so they are set to 1e-10. Functions could (and should) be adapted in further versions to be able to deal 
        with entries = 0"""
        row_sums = torch.sum(trans, dim=1)  # normalize transition matrix, every row should sum up to 1
        row_sums = row_sums[:, None]
        normalized_trans_matr = torch.div(trans, row_sums)
        if not self.FMMIE:  # in this case, the loss only requires the T1 matrix from the Viterbi algorithm
            res = Viterbi(normalized_trans_matr, data, alpha=self.alpha, logscale=True, return_log=True,
                          print_info=False, softmax=self.softmax)
            if self.use_normalized:
                res_normalized = torch.exp(res.T1)  # to normalize data, it has to be transformed back from log scale
                row_sums = torch.sum(res_normalized, dim=0)  # normalize transition matrix
                row_sums = row_sums[None:, ]
                res_normalized = torch.div(res_normalized, row_sums)
                res_normalized = torch.clamp(res_normalized, min=float(1e-10))
                res_normalized = torch.log(res_normalized)
                if self.print_info:
                    print(
                        "[INFO]: Normalization is not correctly implemented. Optimization will not work or will be computationally heavy.")

            else:
                res_normalized = None
            if self.softmax:  # evaluate the softmax result compared to the argmax result
                print(
                    f"Percentage of data where x == y: {np.sum(res.x.detach().cpu().numpy() == np.round((res.y.detach().cpu().numpy()))) / len(res.x)} ")
            return res.x, res.T1, res_normalized, res.y
        else:  # if FMMIE loss function is used
            res = Viterbi(normalized_trans_matr, data, alpha=self.alpha, logscale=True, return_log=True,
                          print_info=False, softmax=self.softmax, FMMIE=True, k_best=self.k_best, labels=targets)
            return res.x, res.res_FMMIE

    def train(self, epoch):
        # initialize variables
        nr, total_loss, total_acc = 0, 0, 0
        cnt_print = 0
        loss = 0.0
        count_elements = 0
        all_targets = []
        all_labels_predicted = []
        for i, (inputs, targets) in enumerate(self.train_loader):
            try:
                temp_length = len(targets[0])  # in case of a defined max length, the current length is saved
            except:
                continue
            if temp_length > 1:
                all_targets.extend(targets[0].tolist())
            if temp_length == 1:
                all_targets.append(targets.item())
            inputs = torch.squeeze(inputs, dim=0)
            targets = torch.squeeze(targets, dim=0)
            inputs = inputs.to(device=self.device)  # push inputs and targets to the device
            targets = targets.to(device=self.device)
            count_elements += temp_length

            if self.softmax:
                targets = targets.to(dtype=torch.float64)

            # forward pass
            if not self.FMMIE:
                labels_predicted, y_predicted_unnormalized, y_predicted_normalized, res_softmax = self.forward(inputs)
            else:
                labels_predicted, new_loss = self.forward(inputs, targets)
                loss += new_loss

            # append the predicted labels to a list
            try:
                all_labels_predicted.extend(labels_predicted.tolist())
            except:
                all_labels_predicted.append(labels_predicted.tolist())

            # calculate the total loss
            if not self.FMMIE:
                if self.use_normalized:
                    pred = y_predicted_normalized
                else:
                    pred = y_predicted_unnormalized
                if self.softmax:
                    loss += self.loss(torch.log(res_softmax), targets)
                else:
                    loss += self.loss(torch.transpose(pred, 0, 1), targets)

            if torch.isnan(loss):  # if loss is NaN, training is stopped
                if self.print_info:
                    print("[INFO]: loss is NaN, training was stopped. Epoch:" + str(epoch))
                return False

            # backward pass and optimization
            if count_elements >= self.min_length:
                # calculate gradients = backward pass
                loss.backward()

                # update weights
                self.optimizer.step()

                # zero the gradients after updating
                self.optimizer.zero_grad()

                # evaluation of the training results (loss and accuracy)
                if (
                        epoch % 1 == 0 or epoch == self.num_epochs - 1) and self.print_results:  # ergibt diese Zeile einen Sinn?
                    total_acc += (np.array(all_labels_predicted) == np.array(all_targets)).sum().item() / len(
                        all_labels_predicted)
                    total_loss += loss.item() / len(all_labels_predicted)
                    nr += 1
                    if cnt_print % 10 == 0:
                        rel_total_acc = total_acc / nr
                        rel_total_loss = total_loss / nr
                        if self.print_results:
                            if torch.is_tensor(self.alpha):
                                alpha = self.alpha.item()
                            else:
                                alpha = self.alpha
                            print(f"Epoch: {epoch}, i = {i + 1}, Alpha = {alpha:.5f}  \nTrain Accuracy (Average): "
                                  f"\t{(100 * rel_total_acc):0.5f}%\tTrain loss (Average): \t{rel_total_loss:.7f}")
                    cnt_print += 1
                loss = 0.0
                count_elements = 0
                all_labels_predicted = []
                all_targets = []

        return True

    def test(self, epoch):
        if epoch % 1 == 0 or epoch == self.num_epochs - 1:  # can be adapted if test results should be printed less often
            test_loss, correct, nr = 0, 0, 0

            with torch.no_grad():
                for i, (inputs, targets) in enumerate(self.test_loader):
                    targets = targets.to(device=self.device)
                    inputs = torch.squeeze(inputs, dim=0)
                    targets = torch.squeeze(targets, dim=0)
                    if self.FMMIE:
                        try:
                            len(targets)
                        except:
                            continue
                    inputs = inputs.to(device=self.device)
                    if self.FMMIE:
                        labels_predicted, loss = self.forward(inputs, targets)
                    else:
                        labels_predicted, y_predicted_unnormalized, y_predicted_normalized, res_softmax = self.forward(
                            inputs)

                    labels_predicted = labels_predicted.to(dtype=torch.int64)

                    # calculate the total loss and accuracy of the test set
                    if not self.FMMIE:
                        if self.use_normalized:
                            pred = y_predicted_normalized
                        else:
                            pred = y_predicted_unnormalized
                        if self.softmax:
                            loss = self.loss(torch.log(res_softmax), targets)
                        else:
                            loss = self.loss(torch.transpose(pred, 0, 1), targets)
                    test_loss += loss.item() / len(labels_predicted)
                    correct += (labels_predicted == targets).sum().item() / len(labels_predicted)
                    nr += 1

                test_loss /= nr
                correct /= nr

                if self.print_results:
                    print(f"Test Accuracy (Average): \t{(100 * correct):0.5f}%\tTest loss (Average): \t{test_loss:.6f}")

    def training(self):
        if self.print_info:
            print("[INFO]: Data has been loaded. Training starts.")
        for epoch in range(self.num_epochs):
            successful = self.train(epoch)  # train the model
            if not successful:
                if self.save_unsuccesful:
                    self.no_nan = False
                    continue  # training is continued, but with the info that loss was NaN at least once
                else:
                    return False  # if loss is NaN, training is stopped and the transition matrix is not saved
            self.test(epoch)  # test the model

        return True

    def save(self, save_unsuccessful):

        trans = torch.clamp(self.trans, min=float(1e-10))  # normalization according to the forward pass
        row_sums = torch.sum(trans, dim=1)
        row_sums = row_sums[:, None]
        trans = torch.div(trans, row_sums)
        trans = trans.detach().cpu().numpy()

        if torch.is_tensor(self.alpha):
            self.alpha = self.alpha.detach().cpu().numpy()  # convert to numpy
        trans = np.append(np.ones((1, 5)) * self.alpha, trans, axis=0)

        # the out_name should be unique for every parameter setting to allow separate storage of the transition matrices
        out_name = ("./Transition_Matrix/optimized_" + self.dataset + "_fold_" + str(self.fold) + "_checkpoints_" +
                    str(self.checkpoints) + "_lr_" + str(self.learning_rate) + "_otrans_" + str(
                    self.train_transition) + "_oalpha_" + str(
                    self.train_alpha) + "_epochs_" + str(self.num_epochs) + "_FMMIE_" + str(
                    self.FMMIE) + "_length_" + str(self.length) + "_trwval_" + str(
                    self.train_with_val)) + "_startalpha_" + str(self.save_alpha)

        if self.successful and self.no_nan and not save_unsuccessful:

            out_name += ".txt"
            np.savetxt(out_name, trans, fmt="%.15f", delimiter=",")
            if self.print_info:
                print("[INFO]: Training of fold " + str(self.fold) + " of the dataset '" + self.dataset +
                      "' was successful. Data has been saved")
        else:
            if save_unsuccessful:
                out_name += "_unsuccessful.txt"
                np.savetxt(out_name, trans, fmt="%.15f", delimiter=",")
                if self.print_info:
                    if self.successful and self.no_nan:
                        print("[INFO]: Training of fold " + str(self.fold) + " of the dataset '" + self.dataset +
                              "' was successful. Data has been saved (However, file name ends with '_unsuccessful')")
                    else:
                        print("[INFO]: training of fold " + str(self.fold) + " was not completed due to NaN. However, "
                                                                             "Transitionmatrix has been saved")
            else:
                if self.print_info:
                    print("[INFO]: training was not successful. Transition matrix will not be saved.")


def main():
    """
    The parameters for training alpha and the transition matrix can be input via arguments or directly here
    """
    try:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--lower_alpha', type=float, default=0.1, help='random seed')
        parser.add_argument('--num_epochs', type=int, default=100, help='random seed')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='random seed')
        parser.add_argument('--train_alpha', type=int, default=1, help='random seed')
        parser.add_argument('--train_transition', type=int, default=0, help='random seed')
        args = parser.parse_args()

        lower_alpha = args.lower_alpha
        num_epochs = args.num_epochs
        learning_rate = args.learning_rate
        if args.train_alpha == 1:
            train_alpha = True
        else:
            train_alpha = False
        if args.train_transition == 1:
            train_transition = True
        else:
            train_transition = False

    except:
        print("Error: No config given via console")
        num_epochs = 100
        learning_rate = 0.001
        train_alpha = True
        train_transition = False
        lower_alpha = 0.1

    for alpha in [lower_alpha, 1.0]:
        for fold in range(1, 11):
            print(
                f'Optimization for alpha: {alpha}, fold: {fold}, learning rate: {learning_rate}, epochs: {num_epochs}, train_alpha: {train_alpha}, train_transition: {train_transition}, train with val True, min_length = 200, ')
            Train_HMM(dataset='Sleep-EDF-2018', num_epochs=num_epochs, learning_rate=learning_rate, print_results=True,
                      train_alpha=train_alpha, train_transition=train_transition, alpha=alpha, fold=fold, save=True,
                      save_unsuccesful=False, use_normalized=False, softmax=False, FMMIE=True, train_with_val=True,
                      min_length=200, length=2000, k_best=20)


if __name__ == "__main__":
    main()
