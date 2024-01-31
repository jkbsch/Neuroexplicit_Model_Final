from Viterbi_Algorithm import *
from HMM_utils import *
import torch


class DnnAndVit:

    def __init__(self, dataset="Sleep-EDF-2018", fold=1, nr=0, used_set="train", trans_matr="EDF-2018",
                 length=None, start=None, logscale=True, alpha=None, print_info=True, checkpoints='given', softmax=False, k_best=1, is_torch=False):
        """
        This class is used to combine the predictions of the DNN and the Viterbi algorithm, thus creating the hybrid predictions
        :param dataset: Name of the dataset
        :param fold: Fold number
        :param nr: Number of the night
        :param used_set: Set used for the DNN (train, val, test)
        :param trans_matr: Transition matrix used for the Viterbi algorithm
        :param length: Length available for the Viterbi algorithm
        :param start: Start index for the Viterbi algorithm
        :param logscale: Whether logscale is used in the Viterbi algorithm
        :param alpha: global parameter Alpha used in the Viterbi algorithm
        :param print_info: Whether information is printed
        :param checkpoints: Checkpoint folder used for the DNN, either 'given' or 'own'
        :param softmax: Whether softmax is used in the Viterbi algorithm
        :param k_best: Number of the best paths returned by the Viterbi algorithm
        :param is_torch: Whether the input data is a torch tensor
        """

        self.P_Matrix_labels = None
        self.P_Matrix_probs = None
        self.start = None
        self.length = length
        self.dataset = dataset
        self.fold = fold
        self.nr = nr
        self.used_set = used_set
        self.trans_matr = trans_matr
        self.start = start
        self.logscale = logscale
        self.alpha = alpha
        self.print_info = print_info
        self.checkpoints = checkpoints
        self.softmax = softmax
        self.k_best = k_best


        # if only the name of the transition matrix is given (as a string), the transition matrix is loaded
        if type(trans_matr) == str:
            self.Transition_Matrix = load_Transition_Matrix(trans_matr)[1]
        else:
            self.Transition_Matrix = trans_matr


        self.P_Matrix_labels, self.P_Matrix_probs = load_P_Matrix(checkpoints, dataset, used_set, fold, nr, print_info)
        self.pure_predictions = pure_predictions(self.P_Matrix_probs)

        self.set_length(length) # sets the length of the Viterbi algorithm, and checks for faulty inputs
        self.set_start(start) # sets the start index of the Viterbi algorithm, and checks for faulty inputs

        self.selected_P_Matrix() # with the length and start index, the P_Matrix is cut to the right size


        # the transition matrix is transformed to a torch tensor if necessary, then the hybrid_predictions function is called
        if torch.is_tensor(self.Transition_Matrix):
            self.hybrid_predictions, self.hybrid_probs, self.hybrid_softmax = self.hybrid_predictions()
        elif is_torch:
            self.Transition_Matrix = torch.from_numpy(self.Transition_Matrix)
            self.P_Matrix_probs = torch.from_numpy(self.P_Matrix_probs)
            if self.alpha is not None:
                self.alpha = torch.from_numpy(self.alpha)

            self.hybrid_predictions, self.hybrid_probs, self.hybrid_softmax = self.hybrid_predictions()
        else:
            self.hybrid_predictions, self.hybrid_probs = self.hybrid_predictions()

        # results are evaluated and stored
        self.korrekt_SleePy = np.sum(self.P_Matrix_labels == self.pure_predictions)
        if self.k_best == 1:
            self.korrekt_hybrid = np.sum(self.P_Matrix_labels == self.hybrid_predictions)
        else:
            self.korrekt_hybrid = []
            for i in range(self.k_best):
                self.korrekt_hybrid.append(np.sum(self.P_Matrix_labels == self.hybrid_predictions[i]))

        if torch.is_tensor(self.Transition_Matrix) and self.softmax:
            self.compare_softmax_argmax = np.sum(self.hybrid_predictions == np.round((self.hybrid_softmax.detach().cpu().numpy())))



    def set_length(self, length): # sets the length of the Viterbi algorithm, and checks for faulty inputs
        if length is None:
            self.length = len(self.P_Matrix_labels)
        elif (length <= len(self.P_Matrix_labels)) and length > 0:
            self.length = length

        elif length > len(self.P_Matrix_labels):
            length = len(self.P_Matrix_labels)
            if self.print_info:
                print(
                    "[INFO]: Length is higher than data length. Length is set to the length of the input file: " + str(
                        length))
            self.length = length
        else:
            self.length = len(self.P_Matrix_labels)
            if self.print_info:
                print("[INFO]: Length is negative or zero. Length is set to the length of the input file: " + str(
                    self.length))

    def set_start(self, start): # sets the start index of the Viterbi algorithm, and checks for faulty inputs
        if (start is None) and (self.length == len(self.P_Matrix_labels)):
            if self.print_info:
                print("[INFO]: Start index is not defined and length is equal to input length; start index is set to 0")
            self.start = 0
        elif start is None:
            self.start = np.random.randint(0, (len(self.P_Matrix_labels)) - self.length)
        elif start < 0 and self.length == len(self.P_Matrix_labels):
            if self.print_info:
                print(
                    "[INFO]: Start index is negative and the length is equal to input length; Start index is set to 0")
            self.start = 0
        elif start < 0:
            self.start = np.random.randint(0, (len(self.P_Matrix_labels)) - self.length)
        elif start > len(self.P_Matrix_labels) - self.length:
            self.start = len(self.P_Matrix_labels) - self.length
            if self.print_info:
                print(
                    "[INFO]: (Start index + length) is higher than data length. Start index is set to the lowest "
                    "possible value: " + str(self.start))
        else:
            self.start = start

    def selected_P_Matrix(self): # with the length and start index, the P_Matrix is cut to the right size
        beg = self.start
        end = beg + self.length
        self.P_Matrix_labels = self.P_Matrix_labels[beg:end]
        self.P_Matrix_probs = self.P_Matrix_probs[beg:end]
        self.pure_predictions = self.pure_predictions[beg:end]

    def hybrid_predictions(self): # applies the viterbi algorithm
        """vit = Viterbi(A=torch.from_numpy(self.Transition_Matrix), P=torch.from_numpy(self.P_Matrix_probs),
        logscale=self.logscale, alpha=self.alpha, print_info=self.print_info) return vit.x.numpy()"""
        vit = Viterbi(A=self.Transition_Matrix, P=self.P_Matrix_probs, logscale=self.logscale, alpha=self.alpha,
                      print_info=self.print_info, softmax=self.softmax, k_best=self.k_best)
        if torch.is_tensor(self.Transition_Matrix):
            return vit.x, vit.T1, vit.y
        else:
            return vit.x, vit.T1


def main():

    hybrid = DnnAndVit(length=None, start=-20, fold=1, nr=0, used_set='train', logscale=True, alpha=None,
                     checkpoints='given', dataset='Sleep-EDF-2018', trans_matr='EDF-2018', k_best=None, is_torch=False)

    # quick evaluation of the results:
    print("Start: ", hybrid.start, " length: ", hybrid.length)
    print("Labels: \t \t \t", hybrid.P_Matrix_labels, "\nSleePyCo Prediction:", hybrid.pure_predictions,
          "\nHybrid Prediction:\t",
          hybrid.hybrid_predictions)
    if hybrid.k_best == 1:
        print("Korrekt SleePy: ", hybrid.korrekt_SleePy, " Korrekt Hybrid: ", hybrid.korrekt_hybrid)
        print("Korrekt SleePy: ", hybrid.korrekt_SleePy / hybrid.length, "Korrekt Hybrid: ", hybrid.korrekt_hybrid / hybrid.length)
    else:
        for i in range(hybrid.k_best):
            print("Korrekt SleePy: ", hybrid.korrekt_SleePy / hybrid.length)
            print(f'Korrekt Hybrid in {i}-best path:', hybrid.korrekt_hybrid[i] / hybrid.length)


if __name__ == '__main__':
    main()
