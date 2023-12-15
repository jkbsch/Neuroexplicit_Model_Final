from Viterbi.Viterbi_Algorithm import *
from HMM_utils import *


class DnnAndVit:

    def __init__(self, dataset="Sleep-EDF-2013", fold=1, nr=0, used_set="train", trans_matr="edf-2013-and-edf-2018",
                 length=None, start=None, logscale=True, alpha=None, print_info=True, checkpoints='given'):
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

        if type(trans_matr) == str:
            self.Transition_Matrix = load_Transition_Matrix(trans_matr)[1]
        else:
            self.Transition_Matrix = trans_matr


        self.P_Matrix_labels, self.P_Matrix_probs = load_P_Matrix(checkpoints, dataset, used_set, fold, nr, print_info)
        self.pure_predictions = pure_predictions(self.P_Matrix_probs)

        self.set_length(length)
        self.set_start(start)

        self.selected_P_Matrix()

        self.hybrid_predictions, self.hybrid_probs = self.hybrid_predictions()


        self.korrekt_SleePy = np.sum(self.P_Matrix_labels == self.pure_predictions)
        self.korrekt_hybrid = np.sum(self.P_Matrix_labels == self.hybrid_predictions)



    def set_length(self, length):
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

    def set_start(self, start):
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

    def selected_P_Matrix(self):
        beg = self.start
        end = beg + self.length
        self.P_Matrix_labels = self.P_Matrix_labels[beg:end]
        self.P_Matrix_probs = self.P_Matrix_probs[beg:end]
        self.pure_predictions = self.pure_predictions[beg:end]

    def hybrid_predictions(self):
        """vit = Viterbi(A=torch.from_numpy(self.Transition_Matrix), P=torch.from_numpy(self.P_Matrix_probs),
        logscale=self.logscale, alpha=self.alpha, print_info=self.print_info) return vit.x.numpy()"""
        vit = Viterbi(A=self.Transition_Matrix, P=self.P_Matrix_probs, logscale=self.logscale, alpha=self.alpha,
                      print_info=self.print_info)
        return vit.x, vit.T1


def main():
    dnn2 = DnnAndVit(length=None, start=-20, fold=1, nr=0, used_set='train', logscale=True, alpha=None,
                     checkpoints='given', dataset='Sleep-EDF-2013', trans_matr='EDF-2013')
    print("Start: ", dnn2.start, " length: ", dnn2.length)
    print("Labels: \t \t \t", dnn2.P_Matrix_labels, "\nSleePyCo Prediction:", dnn2.pure_predictions,
          "\nHybrid Prediction:\t",
          dnn2.hybrid_predictions)

    print("Korrekt SleePy: ", dnn2.korrekt_SleePy, " Korrekt Hybrid: ", dnn2.korrekt_hybrid)
    print("Korrekt SleePy: ", dnn2.korrekt_SleePy / dnn2.length, "Korrekt Hybrid: ", dnn2.korrekt_hybrid / dnn2.length)


if __name__ == '__main__':
    main()
