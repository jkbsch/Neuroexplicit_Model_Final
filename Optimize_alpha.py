import DNN_and_Vit
import utils
from HMM_utils import *


def set_alphas(start_alpha, end_alpha, step):
    start_alpha = int(round(start_alpha / step))
    end_alpha = int(round(end_alpha / step)) + 1
    return start_alpha, end_alpha, step


class OptimizeAlpha:

    def __init__(self, start_alpha=0.0, end_alpha=1.0, step=0.1, dataset='Sleep-EDF-2013', trans_matrix=None,
                 used_set='train', print_all_results=False, checkpoints='given', optimized=False, evaluate_result = True):

        self.best_correct = 0
        self.length = 1
        self.used_set = used_set
        self.print_all_results = print_all_results
        self.checkpoints = checkpoints
        self.optimized = optimized
        self.evaluate_result = evaluate_result

        self.start_alpha, self.end_alpha, self.step = set_alphas(start_alpha, end_alpha, step)
        self.step = step
        self.dataset, self.trans_matrix, self.end_fold, self.end_nr, self.leave_out = set_dataset(self.used_set,
                                                                                                  dataset, trans_matrix)

        self.alpha = 0
        self.optim()

    def optim(self):
        for alpha in range(self.start_alpha, self.end_alpha):
            sum_correct = 0
            sum_length = 0
            alpha = alpha * self.step

            if self.evaluate_result:
                config = {'dataset': self.dataset,
                          'alpha': alpha,
                          'set': self.used_set}
                pred = []
                labels = []


            for fold in range(1, self.end_fold + 1):

                for nr in range(0, self.end_nr + 1):
                    if (fold, nr) in self.leave_out:
                        continue
                    trans_matrix = load_Transition_Matrix(self.trans_matrix, optimized=self.optimized, fold=fold)
                    dnn_vit = DNN_and_Vit.DnnAndVit(dataset=self.dataset, fold=fold, nr=nr, used_set=self.used_set,
                                                    trans_matr=trans_matrix, alpha=alpha, print_info=False,
                                                    checkpoints=self.checkpoints)
                    sum_correct += dnn_vit.korrekt_hybrid
                    sum_length += dnn_vit.length
                    if self.print_all_results:
                        print("Korrekt SleePy: ", dnn_vit.korrekt_SleePy, " Korrekt Hybrid: ", dnn_vit.korrekt_hybrid)
                        print("Korrekt SleePy: ", dnn_vit.korrekt_SleePy / dnn_vit.length, "Korrekt Hybrid: ",
                              dnn_vit.korrekt_hybrid / dnn_vit.length)
                    if self.evaluate_result:
                        pred.extend(dnn_vit.hybrid_predictions)
                        labels.extend(dnn_vit.P_Matrix_labels)

            if self.evaluate_result:
                if fold == self.end_fold:
                    summarize_result(config=config, save=False, fold=fold, y_pred=pred, y_true=labels)


            if sum_correct > self.best_correct:
                self.alpha = alpha
                self.best_correct = sum_correct
                self.length = sum_length

            print("alpha:", alpha, "best alpha: ", self.alpha, "correct:", sum_correct, "accuracy",
                  sum_correct / sum_length)

        print("best alpha between", self.start_alpha * self.step, "and", (self.end_alpha - 1) * self.step, "is: ",
              self.alpha)
        print("best accuracy:", self.best_correct / self.length)


def main():
    OptimizeAlpha(used_set='train', dataset='Sleep-EDF-2018', start_alpha=0.0, end_alpha=0.5, step=0.5,
                  print_all_results=False, trans_matrix='EDF_2018', optimized=False)


if __name__ == "__main__":
    main()
# warum ist accuracy so hoch? Vielleicht weil nur 30 min wake eingerechnet werden sollen?
# warum ging es, als ich andere Werte für Fold genommen hatte?
