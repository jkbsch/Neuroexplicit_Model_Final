import DNN_and_Vit
import utils
from HMM_utils import *


def set_alphas(start_alpha, end_alpha, step):
    start_alpha = int(round(start_alpha / step))
    end_alpha = int(round(end_alpha / step)) + 1
    return start_alpha, end_alpha, step


class OptimizeAlpha:

    def __init__(self, start_alpha=0.0, end_alpha=10.0, step=0.1, dataset='Sleep-EDF-2013', trans_matrix=None,
                 used_set='train', print_all_results=False, checkpoints='given', oalpha=False, otrans=False, evaluate_result=True,
                 visualize=False, optimize_alpha=True, lr=0.001, epochs=60, successful=True):

        self.best_correct = 0
        self.lr = lr
        self.epochs = epochs
        self.successful = successful
        self.length = 1
        self.used_set = used_set
        self.print_all_results = print_all_results
        self.checkpoints = checkpoints
        self.visualize = visualize
        self.alphas = []
        self.accuracies = []

        self.oalpha = oalpha
        self.all_alphas = []
        self.otrans = otrans
        self.evaluate_result = evaluate_result
        if self.evaluate_result:
            optimize_alpha = True

        self.dataset, self.trans_matrix, self.end_fold, self.end_nr, self.leave_out = set_dataset(self.used_set,
                                                                                                  dataset, trans_matrix)
        if self.oalpha and start_alpha != end_alpha:
            print("[INFO]: optimized alpha calculated from training algorithm is used. Alpha will not be optimized")
            start_alpha = end_alpha

        if optimize_alpha:
            self.start_alpha, self.end_alpha, self.step = set_alphas(start_alpha, end_alpha, step)
            self.step = step
            self.alpha = 0
            self.optim()

        if self.visualize:
            self.plot(start_alpha, end_alpha)

    def optim(self):
        for alpha in range(self.start_alpha, self.end_alpha):
            sum_correct = 0
            sum_length = 0
            alpha = alpha * self.step



            if self.evaluate_result:
                pred = []
                labels = []
                sizes = []
                errors_long_hybrid = np.zeros(5, dtype=int)
                errors_long_sleepy = np.zeros(5, dtype=int)
                errors_single_hybrid = np.zeros(5, dtype=int)
                errors_single_sleepy = np.zeros(5, dtype=int)
                errors_fast_changing_hybrid = np.zeros(5, dtype=int)
                errors_fast_changing_sleepy = np.zeros(5, dtype=int)
                nr_long = 0
                nr_single = 0
                nr_fast = 0

                config = {'dataset': self.dataset,
                          'transmatrix': self.trans_matrix,
                          'oalpha': self.oalpha,
                          'otrans': self.otrans,
                          'lr': self.lr,
                          'successful': self.successful,
                          'epochs': self.epochs,
                          'alpha': alpha,
                          'all_alphas': self.all_alphas,
                          'set': self.used_set,
                          'checkpoints': self.checkpoints,
                          'sizes': sizes}

            # self.end_fold = 1
            for fold in range(1, self.end_fold + 1):

                for nr in range(0, self.end_nr + 1):
                    if (fold, nr) in self.leave_out:
                        continue
                    new_alpha, trans_matrix = load_Transition_Matrix(self.trans_matrix, oalpha=self.oalpha,
                                                                     otrans=self.otrans, fold=fold,
                                                                     alpha=alpha, lr=self.lr,
                                                                     successful=self.successful,
                                                                     epochs=self.epochs)
                    if nr == 0:
                        self.all_alphas.append(new_alpha)
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
                        config['sizes'].append(len(dnn_vit.P_Matrix_labels))
                        pred.extend(dnn_vit.hybrid_predictions)
                        labels.extend(dnn_vit.P_Matrix_labels)
                        elh, els, esh, ess, nrl, nrs, efch, efcs, nrf = analyze_errors(y_true=dnn_vit.P_Matrix_labels, sleepy_pred=dnn_vit.pure_predictions, hybrid_pred=dnn_vit.hybrid_predictions)
                        errors_long_hybrid += elh
                        errors_long_sleepy += els
                        errors_single_hybrid += esh
                        errors_single_sleepy += ess
                        nr_long +=nrl
                        nr_single += nrs
                        errors_fast_changing_hybrid += efch
                        errors_fast_changing_sleepy += efcs
                        nr_fast += nrf

            if self.evaluate_result:
                summarize_result(config=config, save=False, fold=fold, y_pred=pred, y_true=labels)

                print(f'Long sleep phases according to labels with a wrong prediction: \n "Pure":\t'
                      f'{errors_long_sleepy}\n "Hybrid":\t{errors_long_hybrid} \n\nLong sleep phases according to '
                      f'label, with the middle sleep phase being different \n(this occured {nr_single} times) with a '
                      f'wrong prediction in the middle: \n "Pure":\t'
                      f'{errors_single_sleepy}\n "Hybrid":\t{errors_single_hybrid}\n\nWrong predictions in a phase '
                      f'of 20 sleep stages where the sleep stages \nchange at least 4 times according to labels (this occured {nr_fast} times): \n '
                      f'"Pure:"\t{errors_fast_changing_sleepy}\n "Hybrid:"\t{errors_fast_changing_hybrid}')

            if sum_correct > self.best_correct:
                self.alpha = alpha
                self.best_correct = sum_correct
                self.length = sum_length

            """print("alpha:", alpha, "best alpha: ", self.alpha, "correct:", sum_correct, "accuracy",
                  sum_correct / sum_length)"""
            self.alphas.append(alpha)
            self.accuracies.append((sum_correct/sum_length)*100)
            print(f'alpha: {alpha:.2f} best alpha: {self.alpha:.2f} correct: {sum_correct} accuracy: {(sum_correct / sum_length)*100:.4f}%')

        """print("best alpha between", self.start_alpha * self.step, "and", (self.end_alpha - 1) * self.step, "is: ",
              self.alpha)"""
        print(f'best alpha between {self.start_alpha*self.step:.2f} and {(self.end_alpha - 1)*self.step:.2f} is {self.alpha:.2f}')
        print(f'best accuracy: {(self.best_correct / self.length)*100:.4f}%')

    def plot(self, start_alpha, end_alpha):
        config = {
            'fold': 1,
            'nr': 0,
            'trans_matrix': self.trans_matrix,
            'oalpha': self.oalpha,
            'otrans': self.otrans,
            'lr': self.lr,
            'successful': self.successful,
            'epochs': self.epochs,
            'all_alphas': self.all_alphas,
            'dataset': self.dataset,
            'alpha': (start_alpha + end_alpha) / 2, ### !!! GGF. ist das hier falsch, wenn mit optimiertem Alpha gearbeitet wird - das ist noch zu sehen
            'used_set': self.used_set,
            'checkpoints': self.checkpoints
        }
        alpha, trans_matrix = load_Transition_Matrix(config['trans_matrix'], oalpha=config['oalpha'], otrans=config["otrans"], lr=config["lr"], fold=config['fold'], epochs=config['epochs'], successful=config['successful'])
        if alpha is not None:
            config["alpha"] = alpha
        dnn_vit = DNN_and_Vit.DnnAndVit(dataset=config['dataset'], fold=config['fold'], nr=config['nr'], used_set=config['used_set'],
                                        trans_matr=trans_matrix, alpha=config['alpha'], print_info=False,
                                        checkpoints=config['checkpoints'])
        y_true = dnn_vit.P_Matrix_labels
        y_pred_sleepy = dnn_vit.pure_predictions
        probs_sleepy = dnn_vit.P_Matrix_probs
        y_pred_hybrid = dnn_vit.hybrid_predictions
        probs_hybrid = dnn_vit.hybrid_probs
        if dnn_vit.logscale:
            probs_hybrid = np.exp(probs_hybrid)
        probs_hybrid = np.divide(probs_hybrid, np.sum(probs_hybrid, axis=0)).T

        posteriogram(y_true, y_pred_hybrid,y_pred_sleepy, config)
        visualize_probs(y_true, probs_hybrid, probs_sleepy, y_pred_sleepy, y_pred_hybrid, config)


def main():
    # visualize_alphas()
    optimize_alpha = OptimizeAlpha(used_set='test', dataset='Sleep-EDF-2018', start_alpha=0.3, end_alpha=0.3, step=0.05,
                                   print_all_results=False, trans_matrix=None, otrans=True, oalpha=False,
                                   evaluate_result=True, visualize=False,
                                   optimize_alpha=False, lr=0.0001, successful=False, epochs=60, checkpoints='given')
    """alphas = []
    accuracies = []
    for dataset in ['Sleep-EDF-2013', 'Sleep-EDF-2018']:
        for used_set in ['train', 'test']:
            print(f'Dataset: {dataset}, used_set: {used_set}')
            optimize_alpha = OptimizeAlpha(used_set=used_set, dataset=dataset, start_alpha=0.0, end_alpha=1.0, step=0.05,
                          print_all_results=False, trans_matrix=None, otrans=False, oalpha=False, evaluate_result=False, visualize=False,
                          optimize_alpha=True, lr=0.0001, successful=False, epochs=60, checkpoints='given')
            alphas.append(optimize_alpha.alphas)
            accuracies.append(optimize_alpha.accuracies)
    alphas = np.array(alphas)
    accuracies = np.array(accuracies)
    np.savetxt("results/alphas_notrain.txt", alphas, fmt="%.15f", delimiter=",")
    np.savetxt("results/accuracies_notrain.txt", accuracies, fmt="%.15f", delimiter=",")"""


if __name__ == "__main__":
    main()
# warum ist accuracy so hoch? Vielleicht weil nur 30 min wake eingerechnet werden sollen?
# warum ging es, als ich andere Werte für Fold genommen hatte?
