import DNN_and_Vit


def set_alphas(start_alpha, end_alpha, step):
    start_alpha = int(start_alpha / step)
    end_alpha = int(end_alpha / step) +1
    return start_alpha, end_alpha, step


class OptimizeAlpha:

    def __init__(self, start_alpha=0, end_alpha=1, step=0.1, dataset='Sleep-EDF-2013', trans_matrix=None,
                 used_set='train', print_all_results=False):

        self.best_correct = 0
        self.length = 1
        self.end_fold = None
        self.end_nr = None
        self.used_set = used_set
        self.print_all_results = print_all_results

        self.start_alpha, self.end_alpha, self.step = set_alphas(start_alpha, end_alpha, step)
        self.step = step
        self.dataset, self.trans_matrix = self.set_dataset(dataset, trans_matrix)

        self.alpha = 0

        self.optim()

    def set_dataset(self, dataset, trans_matrix):
        if dataset == 'Sleep-EDF-2013':
            if self.used_set == 'test':
                self.end_fold = 20
                self.end_nr = 1
            else:
                self.end_fold = 20
                self.end_nr = 32
        elif dataset == 'Sleep-EDF-2018':
            if self.used_set == 'test':
                self.end_fold = 10
                self.end_nr = 15
            else:
                self.end_fold = 10
                self.end_nr = 122
        else:
            print("[INFO]: Dataset is not or incorrectly defined. Dataset is set to Sleep EDF 2013")
            dataset = 'Sleep-EDF-2013'
            if self.used_set == 'test':
                self.end_fold = 20
                self.end_nr = 1
            else:
                self.end_fold = 20
                self.end_nr = 32

        if trans_matrix == 'EDF_2013' or 'EDF_2018' or 'edf-2013-and-edf-2018':
            return dataset, trans_matrix
        elif trans_matrix is not None:
            print("transmatrix was not defined. It is set to the one according to the dataset")

        if dataset == 'Sleep-EDF-2013':
            return dataset, 'EDF_2013'
        elif dataset == 'Sleep-EDF-2018':
            return dataset, 'EDF_2018'
        return None, None

    def optim(self):
        for alpha in range(self.start_alpha, self.end_alpha):
            sum_correct = 0
            sum_length = 0
            alpha = alpha * self.step

            for fold in range(1, self.end_fold+1):
                for nr in range(1, self.end_nr+1):
                    dnn_vit = DNN_and_Vit.DnnAndVit(start=0, length=None, dataset="Sleep-EDF-2013", fold=fold, nr=nr,
                                                    used_set="train", trans_matr="EDF_2013", alpha=alpha, infos=False)
                    sum_correct += dnn_vit.korrekt_hybrid
                    sum_length += dnn_vit.length
                    if self.print_all_results:
                        print("Korrekt SleePy: ", dnn_vit.korrekt_SleePy, " Korrekt Hybrid: ", dnn_vit.korrekt_hybrid)
                        print("Korrekt SleePy: ", dnn_vit.korrekt_SleePy / dnn_vit.length, "Korrekt Hybrid: ",
                              dnn_vit.korrekt_hybrid / dnn_vit.length)

            if sum_correct > self.best_correct:
                self.alpha = alpha
                self.best_correct = sum_correct
                self.length = sum_length

            print("alpha:", alpha, "best alpha: ", self.alpha, "correct:", sum_correct, "accuracy",
                  sum_correct / sum_length)

        print("best alpha:", self.alpha)
        print("best accuracy:", self.best_correct / self.length)


def main():
    OptimizeAlpha(used_set='test', dataset='Sleep-EDF-2018', start_alpha=0.1, end_alpha=0.1,print_all_results=True, trans_matrix='edf-2013-and-edf-2018')


if __name__ == "__main__":
    main()
