import numpy as np
import os
import sklearn.metrics as skmet
from terminaltables import SingleTable
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# loads the transition matrix. If check = True, it only checks whether the name of the transition matrix is correct.
def load_Transition_Matrix(trans_matr="EDF-2018", oalpha=False, otrans=False, fold=1, check=False,
                           successful=True, checkpoints='given', lr=0.00001, alpha=0.3, epochs=60, FMMIE=None, mlength=None, trwval=None, startalpha=None):
    """loads the transition matrix. If check = True, it only checks whether the name of the transition matrix is correct.
    If other datasets are used, make sure to adjust this function accordingly
    """

    # checks different ways of defining the transition matrix
    if (trans_matr == "edf-2013-and-edf-2018" or trans_matr == 'EDF 2013 and 2018' or trans_matr == '2013 2018' or
            trans_matr == 'Sleep-EDF-2013-And-Sleep-EDF-2018' or trans_matr == 'edf_2013_and_edf_2018'):
        trans_matr = "Sleep-EDF-2013-And-Sleep-EDF-2018"
    elif (trans_matr == 'EDF_2013' or trans_matr == 'EDF-2013' or trans_matr == 'Sleep-EDF-2013' or
          trans_matr == 'Sleep_EDF_2013'):
        trans_matr = 'Sleep-EDF-2013'
    elif (trans_matr == 'EDF_2018' or trans_matr == 'EDF-2018' or trans_matr == 'Sleep-EDF-2018' or
          trans_matr == 'Sleep_EDF_2018'):
        trans_matr = 'Sleep-EDF-2018'
    else:
        # if the transition matrix is not correctly defined, False is returned
        return False

    if check:
        return True

    if not oalpha and not otrans:
        Trans_path = "./Transition_Matrix/" + trans_matr + ".txt"
    else:
        Trans_path = ("./Transition_Matrix/optimized_" + trans_matr + "_fold_" + str(fold) + "_checkpoints_" +
                      checkpoints + "_lr_" + str(lr) + "_otrans_" + str(otrans) + "_oalpha_" + str(oalpha) +
                      "_epochs_" + str(epochs))
        if FMMIE is not None and FMMIE:
            Trans_path += f"_FMMIE_{FMMIE}_length_{mlength}_trwval_{trwval}"
            if startalpha is not None:
                Trans_path += f"_startalpha_{startalpha}"
        if successful:
            Trans_path += ".txt"
        else:
            Trans_path += "_unsuccessful.txt"

    transitionmatrix = np.loadtxt(Trans_path, delimiter=",")

    if np.shape(transitionmatrix) == (6, 5):
        # if alpha was trained, its value is stored in the first row of the transition matrix. It is removed here
        alpha = transitionmatrix[0, 0]
        transitionmatrix = transitionmatrix[1:, :]
    else:
        alpha = None

    # transitionmatrix[transitionmatrix <= 0] = float(1e-10)
    return alpha, transitionmatrix


def load_P_Matrix(checkpoints='given', dataset='Sleep-EDF-2013', used_set='train', fold=1, nr=0, print_info=True):
    """
    This functions loads the P_Matrix (labels and probabilities) for the given dataset, fold and nr.
    """
    if checkpoints == "given" or checkpoints == "Given":
        chkpt = ""
    elif checkpoints == "own" or checkpoints == "Own":
        chkpt = "Own-"
    else:
        if print_info:
            print("[INFO]: checkpoints must be chosen between 'given' or 'own'. No checkpoints were (correctly) "
                  "chosen. default is set to 'given'")
        chkpt = ""
    P_path = ("./" + chkpt + "Probability_Data/" + chkpt + dataset + "-" + used_set + "/_dataset_" +
              dataset + "_set_" + used_set + "_fold_" + str(fold) + "_nr_" + str(nr))

    P_Matrix_labels = np.loadtxt(P_path + "_labels.txt", delimiter=",", dtype=int)
    P_Matrix_probs = np.loadtxt(P_path + "_probs.txt", delimiter=",")

    return P_Matrix_labels, P_Matrix_probs


def pure_predictions(P_Matrix_probs): # calculates the pure predictions from the probabilities by applying the argmax
    return np.argmax(P_Matrix_probs, axis=1)


def set_dataset(used_set, dataset, trans_matrix):
    """
    This function can be seen as config file. The final fold and nr differ for every dataset and fold, so some fold-number
    combinations have to be excluded. Make sure to adjust this function when using other datasets.
    end_fold: number of total folds for this dataset
    end_nr: number of nights for every fold in this set
    leave_out: fold-number combinations that are excluded
    """
    if dataset == 'Sleep-EDF-2013':
        if used_set == 'test':
            end_fold = 20
            end_nr = 1
            leave_out = [(14, 1)]
        elif used_set == 'train':
            end_fold = 20
            end_nr = 32
            leave_out = [()]
        elif used_set == 'val':
            end_fold = 20
            end_nr = 2
            leave_out = [(5, 13), (6, 13), (7, 13), (9, 13)]
    elif dataset == 'Sleep-EDF-2018':
        if used_set == 'test':
            end_fold = 10
            end_nr = 15
            leave_out = [(2, 15), (5, 15), (7, 15), (9, 14), (9, 15), (10, 14), (10, 15)]
        elif used_set == 'train':
            end_fold = 10
            end_nr = 122
            leave_out = [()]
        elif used_set == 'val':
            end_fold = 10
            end_nr = 13
            leave_out = [(5, 13), (6, 13), (7, 13), (9, 13)]
    else:
        print("[INFO]: Dataset is not or incorrectly defined. Dataset is set to Sleep-EDF-2013")
        dataset = 'Sleep-EDF-2013'
        if used_set == 'test':
            end_fold = 20
            end_nr = 1
            leave_out = [(14,1)]
        elif used_set == 'train':
            end_fold = 20
            end_nr = 32
            leave_out = [()]
        elif used_set == 'val':
            end_fold = 20
            end_nr = 2
            leave_out = [()]
            print("[ATTENTION]: val set must be checked")

    if load_Transition_Matrix(trans_matrix, check=True): # load transition matrix and check for errors
        return dataset, trans_matrix, end_fold, end_nr, leave_out
    elif trans_matrix is not None:
        print("[INFO]: 'transmatrix' was not (correctly) defined. It is set to the one according to the dataset")

    if dataset == 'Sleep-EDF-2013':
        return dataset, 'EDF_2013', end_fold, end_nr, leave_out
    elif dataset == 'Sleep-EDF-2018':
        return dataset, 'EDF_2018', end_fold, end_nr, leave_out
    return None, None, None, None, None


# adapted from SleePyCo
def summarize_result(config, fold, y_true, y_pred, save=True):
    """
    Summarizes the result by calculating the accuracy, macro f1, kappa, precision, recall and f1 for every class, and
    plots the confusion matrix. This function is called by Optimize_Alpha and used when evaluate_result=True.
    In order to properly print the output on Windows with Pycharm, make sure to use the option "Emulate terminal in Output Console"
    in the Run/Debug configurations.
    """

    if config["oalpha"]:
        alpha = config["all_alphas"]
    else:
        alpha = config["alpha"]

    os.makedirs('../results', exist_ok=True) # create results ordner if it does not exist
    result_dict = skmet.classification_report(y_true, y_pred, digits=3, output_dict=True)
    cm = skmet.confusion_matrix(y_true, y_pred)


    accuracy = round(result_dict['accuracy'] * 100, 1)
    macro_f1 = round(result_dict['macro avg']['f1-score'] * 100, 1)
    kappa = round(skmet.cohen_kappa_score(y_true, y_pred), 3)

    wpr = round(result_dict['0']['precision'] * 100, 1)
    wre = round(result_dict['0']['recall'] * 100, 1)
    wf1 = round(result_dict['0']['f1-score'] * 100, 1)

    n1pr = round(result_dict['1']['precision'] * 100, 1)
    n1re = round(result_dict['1']['recall'] * 100, 1)
    n1f1 = round(result_dict['1']['f1-score'] * 100, 1)

    n2pr = round(result_dict['2']['precision'] * 100, 1)
    n2re = round(result_dict['2']['recall'] * 100, 1)
    n2f1 = round(result_dict['2']['f1-score'] * 100, 1)

    n3pr = round(result_dict['3']['precision'] * 100, 1)
    n3re = round(result_dict['3']['recall'] * 100, 1)
    n3f1 = round(result_dict['3']['f1-score'] * 100, 1)

    rpr = round(result_dict['4']['precision'] * 100, 1)
    rre = round(result_dict['4']['recall'] * 100, 1)
    rf1 = round(result_dict['4']['f1-score'] * 100, 1)

    overall_data = [
        ['ACC', 'MF1', '\u03BA'],
        [accuracy, macro_f1, kappa],
    ]
    # show the results for every class
    perclass_data = [
        [colored('A', 'cyan') + '\\' + colored('P', 'green'), 'W', 'N1', 'N2', 'N3', 'R', 'PR', 'RE', 'F1'],
        ['W', cm[0][0], cm[0][1], cm[0][2], cm[0][3], cm[0][4], wpr, wre, wf1],
        ['N1', cm[1][0], cm[1][1], cm[1][2], cm[1][3], cm[1][4], n1pr, n1re, n1f1],
        ['N2', cm[2][0], cm[2][1], cm[2][2], cm[2][3], cm[2][4], n2pr, n2re, n2f1],
        ['N3', cm[3][0], cm[3][1], cm[3][2], cm[3][3], cm[3][4], n3pr, n3re, n3f1],
        ['R', cm[4][0], cm[4][1], cm[4][2], cm[4][3], cm[4][4], rpr, rre, rf1],
    ]

    #plot confusion matrix
    confusion_matrix = np.divide(cm, (np.sum(cm, 1)).reshape(5, 1))
    fig, ax = plt.subplots()
    ax.matshow(confusion_matrix, cmap='Blues')
    ax.set_yticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])
    ax.set_xticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])

    for i in range(5):
        for j in range(5):
            c = confusion_matrix[j, i] * 100
            if c < 60:
                ax.text(i, j, str(f'{c:.1f}%'), va='center', ha='center', color='black')
            else:
                ax.text(i, j, str(f'{c:.1f}%'), va='center', ha='center', color='cornsilk')
    plt.title('Predicted Class', fontsize=10)

    plt.ylabel('Actual Class')
    plt.figtext(0.01, 0.01,f' \nDataset: {config["dataset"]} Transition Matrix: {config["transmatrix"]}, Set: {config["set"]}, trained alpha: {config["oalpha"]}, trained Transition Matrix: {config["otrans"]}, Alpha: {alpha}, \ncheckpoints: {config["checkpoints"]}, epochs: {config["epochs"]}, lr: {config["lr"]}, maxlength: {config["max_length"]}, FMMIE: {config["FMMIE"]}, mlength: {config["mlength"]}, trwval: {config["trwval"]}, startalpha: {config["startalpha"]}', fontsize=6)

    plt.show()
    if type(alpha) != float:
        if len(alpha) > 1:
            alpha = alpha[0]

    fig.savefig(
        f'results/ConfusionMatrix_Ds{config["dataset"][-1]}TM{config["transmatrix"][-1]}{config["set"]}oa{config["oalpha"]:0}ot{config["otrans"]:0}a{alpha:.2f}{config["checkpoints"]}e{config["epochs"]}lr{config["lr"]}maxlen{config["max_length"]}FMMIE{config["FMMIE"]:0}mlen{config["mlength"]}trw{config["trwval"]:0}sa{config["startalpha"]}.png',dpi=1200)

    overall_dt = SingleTable(overall_data, colored('OVERALL RESULT', 'red'))
    perclass_dt = SingleTable(perclass_data, colored('PER-CLASS RESULT', 'red'))

    print('\n[INFO] Evaluation result from fold 1 to {}'.format(fold))
    print(f'\nDataset: "{config["dataset"]}" Transition Matrix: {config["transmatrix"]}, Set: "{config["set"]}", trained alpha: {config["oalpha"]}, trained Transition Matrix: {config["otrans"]}, Alpha: {alpha}, checkpoints: {config["checkpoints"]}, epochs: {config["epochs"]}, lr: {config["lr"]}, maxlength: {config["max_length"]}, FMMIE: {config["FMMIE"]}, mlength: {config["mlength"]}, trwval: {config["trwval"]}, startalpha: {config["startalpha"]}')
    print('\n' + overall_dt.table)
    print('\n' + perclass_dt.table)
    print(colored(' A', 'cyan') + ': Actual Class, ' + colored('P', 'green') + ': Predicted Class' + '\n\n')

    if save:
        with open(os.path.join(f'results/numbers_Ds{config["dataset"][-1]}TM{config["transmatrix"][-1]}{config["set"]}oa{config["oalpha"]:0}ot{config["otrans"]:0}a{alpha:.2f}{config["checkpoints"]}e{config["epochs"]}lr{config["lr"]}maxlen{config["max_length"]}FMMIE{config["FMMIE"]:0}mlen{config["mlength"]}trw{config["trwval"]:0}sa{config["startalpha"]}'+ '.txt'),
                  'w') as f:
            f.write(
                str(fold) + ' ' +
                str(round(result_dict["accuracy"] * 100, 1)) + ' ' +
                str(round(result_dict["macro avg"]["f1-score"] * 100, 1)) + ' ' +
                str(round(kappa, 3)) + ' ' +
                str(round(result_dict["0"]["f1-score"] * 100, 1)) + ' ' +
                str(round(result_dict["1"]["f1-score"] * 100, 1)) + ' ' +
                str(round(result_dict["2"]["f1-score"] * 100, 1)) + ' ' +
                str(round(result_dict["3"]["f1-score"] * 100, 1)) + ' ' +
                str(round(result_dict["4"]["f1-score"] * 100, 1)) + ' '
            )

    return [accuracy, macro_f1, kappa, wpr, wre, wf1, n1pr, n1re, n1f1, n2pr, n2re, n2f1, n3pr, n3re, n3f1, rpr, rre, rf1, confusion_matrix]


def plot_entire_night(y_true, y_pred, sleepy_pred, config, xmin=400, xmax=450):
    # plots the predictions for the entire night and a section (defined by xmin and xmax), comparing the predictions of
    # SleePyCo, the hybrid model and labels
    length = len(y_true)
    X = np.arange(length)
    fig, ax = plt.subplots(2, 1)


    ax[1].set_xlim(xmin, xmax)
    ax[0].set_xlim(0, length)
    y_true = y_true * -1
    y_pred = y_pred * -1
    sleepy_pred = sleepy_pred * -1

    ax[1].scatter(X, y_true, color='black', label='Labels')
    ax[1].scatter(X, np.where(y_true != y_pred, y_pred, None), color='red', label='Wrong Hybrid Predictions')
    ax[1].scatter(X, np.where(y_pred != sleepy_pred, sleepy_pred, None), color='indigo', label='SleePyCo predictions, when different from Hybrid predictions')
    ax[1].set_ylim(-4.5, 0.5)
    ax[1].set_yticks([0, -1, -2, -3, -4], ["W", "N1", "N2", "N3", "REM"])

    ax[0].step(X, y_true, color='black')
    ax[0].scatter(X, np.where(y_true != y_pred, y_pred, None), color='red', s=6)
    ax[0].scatter(X, np.where(y_pred != sleepy_pred, sleepy_pred, None), color='indigo', s=6)
    ax[0].set_ylim(-4.5, 0.5)
    ax[0].set_yticks([0, -1, -2, -3, -4], ["W", "N1", "N2", "N3", "REM"])

    fig.legend(loc='outside right upper', prop={'size': 6})

    if not config["oalpha"]:
        alpha = config["alpha"]
    else:
        alpha = config["all_alphas"][config["fold"] - 1]

    description = f'Predictions for: Dataset: {config["dataset"]} Set: {config["used_set"]} Fold: {config["fold"]} Nr: {config["nr"]} \nTransition Matrix: {config["trans_matrix"]} Alpha: {alpha} trained Transition: {config["otrans"]}, trained Alpha: {config["oalpha"]}, checkpoints: {config["checkpoints"]}, maxlength: {config["max_length"]}, FMMIE: {config["FMMIE"]}, mlength: {config["mlength"]}, trwval: {config["trwval"]}, startalpha: {config["startalpha"]}'
    plt.figtext(0.1, 0.01, description, fontsize=6)
    plt.show()
    fig.savefig(f'results/figure_posteriogram_Ds{config["dataset"][-1]}TM{config["trans_matrix"][-1]}{config["used_set"]}oa{config["oalpha"]:0}ot{config["otrans"]:0}a{alpha:.2f}{config["checkpoints"]}e{config["epochs"]}lr{config["lr"]}maxlen{config["max_length"]}FMMIE{config["FMMIE"]:0}mlen{config["mlength"]}trw{config["trwval"]:0}sa{config["startalpha"]}{xmin, xmax}.png', dpi=1200)


def posteriogram(y_true, probs_hybrid, probs_sleepy, y_pred_sleepy, y_pred_hybrid, config, xmin=400, xmax=440):
    """
    Plots the posteriogram and the delta / T1 matrix for the given section (defined by xmin and xmax) and displays correct and
    incorrect predictions
    """
    fig, ax = plt.subplots(2, 1)

    probs_sleepy = probs_sleepy[xmin:xmax]
    X = np.arange(xmax - xmin)
    y_true = y_true[xmin:xmax]
    probs_hybrid = probs_hybrid[xmin:xmax]
    y_pred_sleepy = y_pred_sleepy[xmin:xmax]
    y_pred_hybrid = y_pred_hybrid[xmin:xmax]

    ax[0].matshow(probs_sleepy.T, label='Probabilities', cmap='Blues')
    ax[0].scatter(X, y_true, color='black', label='Labels', edgecolors='cornsilk')
    ax[0].scatter(X, np.where(y_true != y_pred_sleepy, y_pred_sleepy, None), color='red', label='Wrong Predictions', edgecolors='cornsilk')

    ax[0].set_title('SleePyCo Predictions')
    ax[0].set_yticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])
    A = np.arange(0, xmax - xmin, 5)
    B = A + np.array(xmin)
    ax[0].set_xticks(A, B)

    ax[1].matshow(probs_hybrid.T, label='Probabilites', cmap='Blues')
    ax[1].scatter(X, y_true, color='black', edgecolors='cornsilk')
    ax[1].scatter(X, np.where(y_true != y_pred_hybrid, y_pred_hybrid, None), color='red', edgecolors='cornsilk')

    ax[1].set_title('Hybrid Predictions')
    ax[1].set_yticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])
    ax[1].set_xticks(A, B)

    fig.legend()

    plt.colorbar(cm.ScalarMappable(norm=None, cmap='Blues'), orientation='horizontal', pad=0.2, shrink=0.6,
                 label='Predicted Probabilites')

    if not config["oalpha"]:
        alpha = config["alpha"]
    else:
        alpha = config["all_alphas"][config["fold"] - 1]

    description = f'\n Dataset: {config["dataset"]} Set: {config["used_set"]} Fold: {config["fold"]} Nr: {config["nr"]} Transition Matrix: {config["trans_matrix"]} Alpha: {alpha:.3f}, checkpoints: {config["checkpoints"]}, \n trained Transition: {config["otrans"]}, trained Alpha: {config["oalpha"]}, maxlength: {config["max_length"]}, FMMIE: {config["FMMIE"]}, mlength: {config["mlength"]}, trwval: {config["trwval"]}, startalpha: {config["startalpha"]}'
    plt.figtext(0.1, 0.01, description, wrap=True, fontsize=6)
    plt.show()
    fig.savefig(f'results/figure_probs_Ds{config["dataset"][-1]}TM{config["trans_matrix"][-1]}{config["used_set"]}oa{config["oalpha"]:0}ot{config["otrans"]:0}a{alpha:.2f}{config["checkpoints"]}e{config["epochs"]}lr{config["lr"]}maxlen{config["max_length"]}FMMIE{config["FMMIE"]:0}mlen{config["mlength"]}trw{config["trwval"]:0}sa{config["startalpha"]}{xmin, xmax}.png', dpi=1200)

def visualize_alphas():
    """
    This function plots the accuracies for different alphas. The alphas first have to be calculated by the optimize_alpha
    function and then, they have to be stored (saving the results is not implemented in this version).
    Make sure to adjust the input path
    """
    alphas = np.loadtxt('results/new_alphas_notrain_exact_maxlength10_step0.05.txt', delimiter=',')
    accuracies = np.loadtxt('results/new_accuracies_notrain_exact_maxlength10_step0.05.txt', delimiter=',')
    fig, ax = plt.subplots()

    fig.suptitle('Alphas and respective Accuracies, predictions for N = 10')
    ax.scatter(alphas[0], accuracies[0], label='Training Set', color='blue')
    ax.scatter(alphas[1], accuracies[1], label='Test Set', color='mediumpurple')
    ax.scatter(alphas[2], accuracies[2], label='Validation Set', color='indigo')

    plt.grid(True)
    plt.xlim((0, 2))
    plt.ylim((82.5, 92.5))


    plt.xlabel('alpha')
    plt.ylabel('Accuracy in %')
    plt.legend()

    plt.show()

    fig.tight_layout()
    fig.savefig(f'results/new_Comparing alphas_untrained_unlimited_long.png', dpi=1200)

def analyze_errors(y_true, hybrid_pred,sleepy_pred):
    """
    This function analyzes the errors depending on the context in which they were made
    """
    length = len(y_true)
    nr_same_epochs = 7

    # errors_long_hybrid: counts the errors for the hybrid model and SleePyCo in a long and constant phase (defined by nr_same_epochs)
    y_true = np.array(y_true)
    hybrid_pred = np.array(hybrid_pred)
    sleepy_pred = np.array(sleepy_pred)
    errors_long_hybrid = np.zeros(5, dtype=int)
    errors_long_sleepy = np.zeros(5, dtype=int)
    nr_long = 0

    # errors_single_hybrid: counts the errors for the hybrid model in a long and constant phase (defined by nr_same_epochs), if the middle epoch differs
    errors_single_hybrid = np.zeros(5, dtype=int)
    errors_single_sleepy = np.zeros(5, dtype=int)
    nr_single = 0



    for i in range(length-nr_same_epochs):
        arr = y_true[i:i+nr_same_epochs]
        if np.all(arr == arr[0]):
            nr_long += 1
            if hybrid_pred[i+int(nr_same_epochs/2)] != arr[0]:
                errors_long_hybrid[arr[0]] += 1
            if sleepy_pred[i+int(nr_same_epochs/2)] != arr[0]:
                errors_long_sleepy[arr[0]] += 1
        half = int(nr_same_epochs/2)
        arr1 = y_true[i:i+half]
        arr2 = y_true[i+half+1:i+nr_same_epochs]
        middle = y_true[i+half]

        if np.all(arr1 == arr1[0]) and np.all(arr2 == arr2[0]) and arr1[0] == arr2[0] and middle != arr1[0]:
            nr_single += 1
            if hybrid_pred[i+half] != middle:
                errors_single_hybrid[middle] += 1
            if sleepy_pred[i+half] != middle:
                errors_single_sleepy[middle] += 1

    errors_fast_changing_hybrid = np.zeros(5, dtype=int)
    errors_fast_changing_sleepy = np.zeros(5, dtype=int)
    nr_fast = 0

    nr_changing = 20
    threshold = 4

    # errors_fast_changing counts the number of errors SleePyCo / the hybrid model makes in a phase of length
    # nr_changing, if the number of changes is higher than threshold

    for i in range(length-nr_changing):
        count_changes = 0
        for j in range(nr_changing-1):
            if(y_true[i+j] != y_true[i+j+1]):
                count_changes += 1
        if count_changes >= threshold:
            nr_fast += 1
            if hybrid_pred[i+int(nr_changing/2)] != y_true[i+int(nr_changing/2)]:
                errors_fast_changing_hybrid[y_true[i+int(nr_changing/2)]] += 1
            if sleepy_pred[i+int(nr_changing/2)] != y_true[i+int(nr_changing/2)]:
                errors_fast_changing_sleepy[y_true[i+int(nr_changing/2)]] += 1



    return errors_long_hybrid, errors_long_sleepy, errors_single_hybrid, errors_single_sleepy, nr_long, nr_single, errors_fast_changing_hybrid, errors_fast_changing_sleepy, nr_fast

