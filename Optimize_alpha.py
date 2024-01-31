import DNN_and_Vit
from HMM_utils import *


def set_alphas(start_alpha, end_alpha, step):
    start_alpha = int(round(start_alpha / step))
    end_alpha = int(round(end_alpha / step)) + 1
    return start_alpha, end_alpha, step


class OptimizeAlpha:

    def __init__(self, start_alpha=0.0, end_alpha=10.0, step=0.1, dataset='Sleep-EDF-2018', trans_matrix=None,optimize_alpha=True,
                 used_set='train', print_all_results=False, checkpoints='given', max_length=None, evaluate_result=True,
                 visualize=False, oalpha=False, otrans=False, lr=0.001, epochs=60, successful=True, FMMIE=None, mlength=None, trwval=None, startalpha=None):
        """
        This function serves multiple purposes, as it is able to evaluate model performance with different alphas,
        create a detailed evaluation of the model performance, and visualize the predictions of the models.
        :param start_alpha: Start value for alpha
        :param end_alpha: End value for alpha
        :param step: Step size between alphas
        :param dataset: Name of the dataset
        :param trans_matrix: Name of the transition matrix; If None, the transition matrix according to the dataset is chosen
        :param optimize_alpha: Whether alpha is optimized; Will be set to true if evaluate_result is true
        :param used_set: Set used for the DNN (train, val, test)
        :param print_all_results: Whether all results are printed while optimizing alpha
        :param checkpoints: Checkpoint folder used for the DNN, either 'given' or 'own'
        :param max_length: Maximum length for the Viterbi algorithm; If None, the entire night is used, else, the night
        is split up into sections of max_length
        :param evaluate_result: Whether a detailed evaluation of the model performance is created, including the creation of the confusion matrix and evaluation in context
        :param visualize: Whether the predictions of the model are visualized, that means the creation of the hypnogram and posteriogram

        :param oalpha: Whether the alpha was trained with backpropagation
        :param otrans: Whether the transition matrix was trained with backpropagation
        Note: if oalpha = False and otrans = False, the following parameters are irrelevant; if not, they are required
        to choose the correct transition matrix
        :param lr: Learning rate used for training
        :param epochs: Number of epochs used for training
        :param successful: Whether the training was successful; (False means that you have allowed to save transition matrices that could
        have been created during training giving a NaN-loss)
        :param FMMIE: Whether the FMMIE loss function was used for training
        :param mlength: Maximum length of a sequence used during training
        :param trwval: Whether the transition matrix / alpha was trained with the validation set (or with the training set)
        :param startalpha: Start value for alpha used during training
        """


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
        self.max_length = max_length
        self.FMMIE = FMMIE
        self.mlength = mlength
        self.trwval = trwval
        self.startalpha = startalpha
        self.res = None

        self.oalpha = oalpha
        self.all_alphas = []
        self.otrans = otrans
        self.evaluate_result = evaluate_result
        if self.evaluate_result:
            optimize_alpha = True

        # set the dataset and transition matrix
        self.dataset, self.trans_matrix, self.end_fold, self.end_nr, self.leave_out = set_dataset(self.used_set,
                                                                                                  dataset, trans_matrix)
        if self.oalpha and start_alpha != end_alpha:
            print("[INFO]: optimized alpha calculated from training algorithm is used. Alpha will not be optimized")
            start_alpha = end_alpha

        # determine the values for alpha and optimize
        if optimize_alpha:
            self.start_alpha, self.end_alpha, self.step = set_alphas(start_alpha, end_alpha, step)
            self.step = step
            self.alpha = 0
            self.optim()

        # visualize the results
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

                # config required to correctly save the results
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
                          'sizes': sizes,
                          'max_length': self.max_length,
                          'FMMIE': self.FMMIE,
                          'mlength': self.mlength,
                          'trwval': self.trwval,
                          'startalpha': self.startalpha}

            # loop through all folds and subjects
            for fold in range(1, self.end_fold + 1):
                for nr in range(0, self.end_nr + 1):
                    if (fold, nr) in self.leave_out:
                        continue
                    new_alpha, trans_matrix = load_Transition_Matrix(self.trans_matrix, oalpha=self.oalpha,
                                                                     otrans=self.otrans, fold=fold,
                                                                     alpha=alpha, lr=self.lr,
                                                                     successful=self.successful,
                                                                     epochs=self.epochs, FMMIE=self.FMMIE, mlength=self.mlength, trwval=self.trwval, startalpha=self.startalpha)
                    if new_alpha is not None and self.oalpha:
                        alpha = new_alpha
                    if nr == 0:
                        self.all_alphas.append(new_alpha)
                    if self.max_length is None or self.max_length == 0: # entire night is evaluated
                        # apply the viterbi algorithm
                        dnn_vit = DNN_and_Vit.DnnAndVit(dataset=self.dataset, fold=fold, nr=nr, used_set=self.used_set, trans_matr=trans_matrix, alpha=alpha, print_info=False, checkpoints=self.checkpoints)
                        correct_hybrid = dnn_vit.korrekt_hybrid
                        correct_SleePy = dnn_vit.korrekt_SleePy
                        hybrid_length = dnn_vit.length
                        P_Matrix_labels = dnn_vit.P_Matrix_labels
                        hybrid_predictions = dnn_vit.hybrid_predictions
                        pure_predictions = dnn_vit.pure_predictions
                    else: # night is split up into sections of max_length
                        length = len(load_P_Matrix(self.checkpoints, self.dataset, self.used_set, fold, nr, False)[0])
                        correct_hybrid = 0
                        hybrid_length = 0
                        correct_SleePy = 0
                        P_Matrix_labels = []
                        hybrid_predictions = []
                        pure_predictions = []

                        for i in range(0, length-self.max_length, self.max_length): # apply the viterbi algorithm
                            dnn_vit = DNN_and_Vit.DnnAndVit(dataset=self.dataset, fold=fold, nr=nr,
                                                            used_set=self.used_set, trans_matr=trans_matrix,
                                                            alpha=alpha, print_info=True, checkpoints=self.checkpoints,
                                                            length=self.max_length, start=i)
                            correct_hybrid += dnn_vit.korrekt_hybrid
                            correct_SleePy += dnn_vit.korrekt_SleePy
                            hybrid_length += dnn_vit.length
                            P_Matrix_labels.extend(dnn_vit.P_Matrix_labels)
                            hybrid_predictions.extend(dnn_vit.hybrid_predictions)
                            pure_predictions.extend(dnn_vit.pure_predictions)

                        mod = length % self.max_length
                        if mod != 0:
                            dnn_vit = DNN_and_Vit.DnnAndVit(dataset=self.dataset, fold=fold, nr=nr, used_set=self.used_set, trans_matr=trans_matrix, alpha=alpha, print_info=True, checkpoints=self.checkpoints, length=mod, start=length - mod)
                            correct_hybrid += dnn_vit.korrekt_hybrid
                            correct_SleePy += dnn_vit.korrekt_SleePy
                            hybrid_length += dnn_vit.length
                            P_Matrix_labels.extend(dnn_vit.P_Matrix_labels)
                            hybrid_predictions.extend(dnn_vit.hybrid_predictions)
                            pure_predictions.extend(dnn_vit.pure_predictions)

                    sum_correct += correct_hybrid
                    sum_length += hybrid_length
                    if self.print_all_results:
                        print("Korrekt SleePy: ", correct_SleePy, " Korrekt Hybrid: ", correct_hybrid)
                        print("Korrekt SleePy: ", correct_SleePy / hybrid_length, "Korrekt Hybrid: ",
                              correct_hybrid / hybrid_length)
                    if self.evaluate_result:
                        config['sizes'].append(len(P_Matrix_labels))
                        pred.extend(hybrid_predictions)
                        labels.extend(P_Matrix_labels)
                        elh, els, esh, ess, nrl, nrs, efch, efcs, nrf = analyze_errors(y_true=P_Matrix_labels, sleepy_pred=pure_predictions, hybrid_pred=hybrid_predictions)
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
                self.res = summarize_result(config=config, save=False, fold=fold, y_pred=pred, y_true=labels) # plot confusion matrix

                print(f'Long sleep phases according to labels with a wrong prediction: \n "Pure":\t'
                      f'{errors_long_sleepy}\n "Hybrid":\t{errors_long_hybrid} \n\nLong sleep phases according to '
                      f'label, with the middle sleep phase being different \n(this occured {nr_single} times) with a '
                      f'wrong prediction in the middle: \n "Pure":\t'
                      f'{errors_single_sleepy}\n "Hybrid":\t{errors_single_hybrid}\n\nWrong predictions in a phase '
                      f'of 20 sleep stages where the sleep stages \nchange at least 4 times according to labels (this occured {nr_fast} times): \n '
                      f'"Pure:"\t{errors_fast_changing_sleepy}\n "Hybrid:"\t{errors_fast_changing_hybrid}')

            if sum_correct > self.best_correct: # compare nr of correct classifications with the best so far
                self.alpha = alpha
                self.best_correct = sum_correct
                self.length = sum_length


            self.alphas.append(alpha)
            self.accuracies.append((sum_correct/sum_length)*100)
            print(f'alpha: {alpha:.2f} best alpha: {self.alpha:.2f} correct: {sum_correct} accuracy: {(sum_correct / sum_length)*100:.4f}%')


        print(f'best alpha between {self.start_alpha*self.step:.2f} and {(self.end_alpha - 1)*self.step:.2f} is {self.alpha:.2f}')
        print(f'best accuracy: {(self.best_correct / self.length)*100:.4f}%')

    def plot(self, start_alpha, end_alpha): # plot the results
        # config to save the results and to create the caption
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
            'alpha': (start_alpha + end_alpha) / 2,
            'used_set': self.used_set,
            'checkpoints': self.checkpoints,
            'max_length': self.max_length,
            'FMMIE': self.FMMIE,
            'mlength': self.mlength,
            'trwval': self.trwval,
            'startalpha': self.startalpha
        }
        alpha, trans_matrix = load_Transition_Matrix(config['trans_matrix'], oalpha=config['oalpha'], otrans=config["otrans"], lr=config["lr"], fold=config['fold'], epochs=config['epochs'], successful=config['successful'], FMMIE = config['FMMIE'], mlength=config['mlength'], trwval=config['trwval'], startalpha=config['startalpha'])
        if alpha is not None:
            config["alpha"] = alpha
        if config['max_length'] is None or config['max_length'] == 0: # evaluation for entire night
            dnn_vit = DNN_and_Vit.DnnAndVit(dataset=config['dataset'], fold=config['fold'], nr=config['nr'], used_set=config['used_set'],
                                            trans_matr=trans_matrix, alpha=config['alpha'], print_info=False,
                                            checkpoints=config['checkpoints'])
            P_Matrix_labels = dnn_vit.P_Matrix_labels
            hybrid_predictions = dnn_vit.hybrid_predictions
            pure_predictions = dnn_vit.pure_predictions
            P_Matrix_probs = dnn_vit.P_Matrix_probs
            hybrid_probs = dnn_vit.hybrid_probs
            vitlogscale = dnn_vit.logscale

        else: # evaluation for sections of length max_length
            length = len(load_P_Matrix(config['checkpoints'], config['dataset'], config['used_set'], config['fold'], config['nr'], True)[0])
            correct_hybrid = 0
            hybrid_length = 0
            correct_SleePy = 0
            P_Matrix_labels = []
            hybrid_predictions = []
            pure_predictions = []
            P_Matrix_probs = []
            hybrid_probs = []

            # if the evaluation of sections possibly contains more epochs than max_length, therefore several predictions
            # need to be combined
            for i in range(0, length - config['max_length'], config['max_length']):
                dnn_vit = DNN_and_Vit.DnnAndVit(dataset=config['dataset'], fold=config['fold'], nr=config['nr'], used_set=config['used_set'],
                                                trans_matr=trans_matrix, alpha=config['alpha'], print_info=False,
                                                checkpoints=config['checkpoints'], length=config['max_length'], start=i)
                correct_hybrid += dnn_vit.korrekt_hybrid
                correct_SleePy += dnn_vit.korrekt_SleePy
                hybrid_length += dnn_vit.length
                P_Matrix_labels.extend(dnn_vit.P_Matrix_labels)
                hybrid_predictions.extend(dnn_vit.hybrid_predictions)
                pure_predictions.extend(dnn_vit.pure_predictions)
                P_Matrix_probs.extend(dnn_vit.P_Matrix_probs)
                hybrid_probs.extend(dnn_vit.hybrid_probs.T)
                vitlogscale = dnn_vit.logscale

            mod = length % config['max_length']
            if mod != 0:
                dnn_vit = DNN_and_Vit.DnnAndVit(dataset=config['dataset'], fold=config['fold'], nr=config['nr'], used_set=config['used_set'],
                                                trans_matr=trans_matrix, alpha=config['alpha'], print_info=False,
                                                checkpoints=config['checkpoints'], length=mod, start=length - mod)
                correct_hybrid += dnn_vit.korrekt_hybrid
                correct_SleePy += dnn_vit.korrekt_SleePy
                hybrid_length += dnn_vit.length
                P_Matrix_labels.extend(dnn_vit.P_Matrix_labels)
                hybrid_predictions.extend(dnn_vit.hybrid_predictions)
                pure_predictions.extend(dnn_vit.pure_predictions)
                P_Matrix_probs.extend(dnn_vit.P_Matrix_probs)
                hybrid_probs.extend(dnn_vit.hybrid_probs.T)
            hybrid_probs = np.array(hybrid_probs).T

        y_true = np.array(P_Matrix_labels)
        y_pred_sleepy = np.array(pure_predictions)
        probs_sleepy = np.array(P_Matrix_probs)
        y_pred_hybrid = np.array(hybrid_predictions)
        probs_hybrid = np.array(hybrid_probs)
        if vitlogscale:
            probs_hybrid = np.exp(probs_hybrid)
        probs_hybrid = np.divide(probs_hybrid, np.sum(probs_hybrid, axis=0)).T

        # xmin and xmax are set here
        xmin=400
        xmax = 450
        plot_entire_night(y_true, y_pred_hybrid, y_pred_sleepy, config, xmin, xmax)
        xmin=410
        xmax = 450
        posteriogram(y_true, probs_hybrid, probs_sleepy, y_pred_sleepy, y_pred_hybrid, config, xmin, xmax)

        # the code below finds especially interesting posteriograms, which analyze the sections where SleePyCo is
        # better than the hybrid model and vice versa
        # find indexes with length 40, where y_pred_hybrid fits best y_true and where y_pred_sleepy fits the data bad

        index_length = 40
        best_sleepy = 0
        best_hybrid = 0
        best_start_index_for_sleepy = 0
        best_start_index_for_hybrid = 0
        for i in range(len(y_true)-index_length-1):
            best_sleepy_new = np.sum(((y_true[i:i + index_length] == y_pred_sleepy[i:i + index_length])*(y_pred_sleepy[i:i + index_length] != y_pred_hybrid[i:i + index_length])))
            if best_sleepy_new > best_sleepy:
                best_sleepy = best_sleepy_new
                best_start_index_for_sleepy = i
            best_hybrid_new = np.sum(((y_true[i:i+index_length] == y_pred_hybrid[i:i+index_length]) * (y_pred_sleepy[i:i+index_length] != y_pred_hybrid[i:i+index_length])))

            if best_hybrid_new > best_hybrid:
                best_hybrid = best_hybrid_new
                best_start_index_for_hybrid = i

        xmin = best_start_index_for_sleepy
        xmax = best_start_index_for_sleepy + index_length
        posteriogram(y_true, probs_hybrid, probs_sleepy, y_pred_sleepy, y_pred_hybrid, config, xmin, xmax)

        xmin = best_start_index_for_hybrid
        xmax = best_start_index_for_hybrid + index_length
        posteriogram(y_true, probs_hybrid, probs_sleepy, y_pred_sleepy, y_pred_hybrid, config, xmin, xmax)

def main():
    optimize_alpha = OptimizeAlpha(used_set='train', dataset='Sleep-EDF-2018', start_alpha=0.2, end_alpha=0.2, step=0.05,
                                   print_all_results=False, trans_matrix=None,max_length=None, otrans=True, oalpha=False,
                                   evaluate_result=True, visualize=False,
                                   optimize_alpha=False, lr=0.00001, successful=True, epochs=100, checkpoints='given',
                                FMMIE=True, mlength=1000, trwval=True, startalpha=1.0)





if __name__ == "__main__":
    main()

