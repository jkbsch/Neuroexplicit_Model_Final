import numpy as np
import os
import sklearn.metrics as skmet
from terminaltables import SingleTable
from termcolor import colored

def load_Transition_Matrix(trans_matr="edf-2013-and-edf-2018", optimized=False, fold=1, check=False, successful=True, checkpoints='given'):

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
        return False

    if check:
        return True

    if not optimized:
        Trans_path = "./Transition_Matrix/" + trans_matr + ".txt"
    else:
        if successful:
            Trans_path = ("./Transition_Matrix/optimized_"+trans_matr+"_fold_"+str(fold)+"_checkpoints_"+checkpoints+
                          ".txt")
        else:
            Trans_path = ("./Transition_Matrix/optimized_"+trans_matr+"_fold_"+str(fold)+"_checkpoints_"+checkpoints+
                          "_unsuccessful.txt")

    transitionmatrix = np.loadtxt(Trans_path, delimiter=",")
    return transitionmatrix


def load_P_Matrix(checkpoints='given', dataset='Sleep-EDF-2013', used_set='train', fold=1, nr=0, print_info=True):
    if checkpoints == "given":
        chkpt = ""
    elif checkpoints == "own":
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

def pure_predictions(P_Matrix_probs):
    return np.argmax(P_Matrix_probs, axis=1)


def set_dataset(used_set, dataset, trans_matrix):
    if dataset == 'Sleep-EDF-2013':
        if used_set == 'test':
            end_fold = 20
            end_nr = 1
            leave_out = [(14, 1)]
        else:
            end_fold = 20
            end_nr = 32
            leave_out = [()]
    elif dataset == 'Sleep-EDF-2018':
        if used_set == 'test':
            end_fold = 10
            end_nr = 15
            leave_out = [(2, 15), (5, 15), (7, 15), (9, 14), (9, 15), (10, 14), (10, 15)]
        else:
            end_fold = 10
            end_nr = 122
            leave_out = [()]
    else:
        print("[INFO]: Dataset is not or incorrectly defined. Dataset is set to Sleep-EDF-2013")
        dataset = 'Sleep-EDF-2013'
        if used_set == 'test':
            end_fold = 20
            end_nr = 1
            leave_out = [()]
        else:
            end_fold = 20
            end_nr = 32
            leave_out = [()]

    if load_Transition_Matrix(trans_matrix, check=True):
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
    os.makedirs('results', exist_ok=True)
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

    perclass_data = [
        [colored('A', 'cyan') + '\\' + colored('P', 'green'), 'W', 'N1', 'N2', 'N3', 'R', 'PR', 'RE', 'F1'],
        ['W', cm[0][0], cm[0][1], cm[0][2], cm[0][3], cm[0][4], wpr, wre, wf1],
        ['N1', cm[1][0], cm[1][1], cm[1][2], cm[1][3], cm[1][4], n1pr, n1re, n1f1],
        ['N2', cm[2][0], cm[2][1], cm[2][2], cm[2][3], cm[2][4], n2pr, n2re, n2f1],
        ['N3', cm[3][0], cm[3][1], cm[3][2], cm[3][3], cm[3][4], n3pr, n3re, n3f1],
        ['R', cm[4][0], cm[4][1], cm[4][2], cm[4][3], cm[4][4], rpr, rre, rf1],
    ]

    overall_dt = SingleTable(overall_data, colored('OVERALL RESULT', 'red'))
    perclass_dt = SingleTable(perclass_data, colored('PER-CLASS RESULT', 'red'))

    print('\n[INFO] Evaluation result from fold 1 to {}'.format(fold))
    print(f'\nDataset: "{config["dataset"]}", Set: "{config["set"]}", Alpha: {config["alpha"]}')
    print('\n' + overall_dt.table)
    print('\n' + perclass_dt.table)
    print(colored(' A', 'cyan') + ': Actual Class, ' + colored('P', 'green') + ': Predicted Class' + '\n\n')

    if save:
        with open(os.path.join('results', config['dataset'] +'_alpha_' + config['alpha']+':_set_'+ config['set'] + '.txt'), 'w') as f:
            f.write(
                str(fold) + ' ' +
                str(round(result_dict['accuracy'] * 100, 1)) + ' ' +
                str(round(result_dict['macro avg']['f1-score'] * 100, 1)) + ' ' +
                str(round(kappa, 3)) + ' ' +
                str(round(result_dict['0.0']['f1-score'] * 100, 1)) + ' ' +
                str(round(result_dict['1.0']['f1-score'] * 100, 1)) + ' ' +
                str(round(result_dict['2.0']['f1-score'] * 100, 1)) + ' ' +
                str(round(result_dict['3.0']['f1-score'] * 100, 1)) + ' ' +
                str(round(result_dict['4.0']['f1-score'] * 100, 1)) + ' '
            )
