from Viterbi.Viterbi_Algorithm import *


class DnnAndVit:

    def __init__(self, dataset="Sleep-EDF-2013", fold=1, nr=0, used_set="train", trans_matr="edf-2013-and-edf-2018",
                 length=None, start=None, logscale = False):
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

        self.Transition_Matrix = self.load_Transition_Matrix()
        self.load_P_Matrix()
        self.pure_predictions = self.pure_predictions()

        self.set_length(length)
        self.set_start(start)

        self.selected_P_Matrix()

        self.hybrid_predictions = self.hybrid_predictions()

    def load_Transition_Matrix(self):
        Trans_path = "./Transition_Matrix/" + self.trans_matr + ".txt"
        a = np.loadtxt(Trans_path, delimiter=",")
        return a

    def load_P_Matrix(self):
        P_path = (
                "./Probability_Data/" + self.dataset + "/_dataset_" + self.dataset + "_set_" + self.used_set + "_fold_" + str(
            self.fold) + "_nr_" + str(self.nr))

        self.P_Matrix_labels = np.loadtxt(P_path + "_labels.txt", delimiter=",", dtype=int)
        self.P_Matrix_probs = np.loadtxt(P_path + "_probs.txt", delimiter=",")

    def pure_predictions(self):
        return np.argmax(self.P_Matrix_probs, axis=1)

    def set_length(self, length):
        if length is None:
            self.length = len(self.P_Matrix_labels)
        elif (length <= len(self.P_Matrix_labels)) and length > 0:
            self.length = length

        elif length > len(self.P_Matrix_labels):
            l = len(self.P_Matrix_labels)
            print("Length is higher than data length. Length is set to the length of the input file: " + str(l))
            self.length = l
        else:
            self.length = len(self.P_Matrix_labels)
            print("Length is negative or zero. Length is set to the length of the input file: " + str(self.length))

    def set_start(self, start):
        if (start is None) and (self.length == len(self.P_Matrix_labels)):
            print("Start index is not defined and length is equal to input length; start index is set to 0")
            self.start = 0
        elif start is None:
            self.start = np.random.randint(0, (len(self.P_Matrix_labels)) - self.length)
        elif start < 0 and self.length == len(self.P_Matrix_labels):
            print("Start index is negative and the length is equal to input length; Start index is set to 0")
            self.start = 0
        elif start < 0:
            self.start = np.random.randint(0, (len(self.P_Matrix_labels)) - self.length)
        elif start > len(self.P_Matrix_labels) - self.length:
            self.start = len(self.P_Matrix_labels) - self.length
            print(
                "(Start index + length) is higher than data length. Start index is set to the lowest possible value: " + str(
                    self.start))
        else:
            self.start = start

    def selected_P_Matrix(self):
        beg = self.start
        end = beg + self.length
        self.P_Matrix_labels = self.P_Matrix_labels[beg:end]
        self.P_Matrix_probs = self.P_Matrix_probs[beg:end]
        self.pure_predictions = self.pure_predictions[beg:end]

    def hybrid_predictions(self):
        vit = Viterbi(A = self.Transition_Matrix, P = self.P_Matrix_probs, logscale=self.logscale)
        result = vit.x
        return result


def main():
    dnn2 = DnnAndVit(length=300, start=800, fold=1, nr=2, logscale=False)
    print("Start: ", dnn2.start, " length: ", dnn2.length)
    print("Labels: \t \t \t", dnn2.P_Matrix_labels, "\nSleePyCo Prediction:", dnn2.pure_predictions, "\nHybrid Prediction:\t",
          dnn2.hybrid_predictions)

    korrekt_SleePy = np.sum(dnn2.P_Matrix_labels == dnn2.pure_predictions)
    korrekt_hybrid = np.sum(dnn2.P_Matrix_labels == dnn2.hybrid_predictions)
    print("Korrekt SleePy: ", korrekt_SleePy, " Korrekt Hybrid: ", korrekt_hybrid)
    print("Korrekt SleePy: ", korrekt_SleePy/dnn2.length, "Korrekt Hybrid: ", korrekt_hybrid/dnn2.length)


if __name__ == '__main__':
    main()
