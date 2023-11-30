import numpy as np


def load_Transition_Matrix(trans_matr):
    Trans_path = "./Transition_Matrix/" + trans_matr + ".txt"
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

    if trans_matrix == 'EDF_2013' or trans_matrix == 'EDF_2018' or trans_matrix == 'edf-2013-and-edf-2018' or trans_matrix == 'first_optimized_trans_matrix':
        return dataset, trans_matrix, end_fold, end_nr, leave_out
    elif trans_matrix is not None:
        print("[INFO]: transmatrix was not (correctly) defined. It is set to the one according to the dataset")

    if dataset == 'Sleep-EDF-2013':
        return dataset, 'EDF_2013', end_fold, end_nr, leave_out
    elif dataset == 'Sleep-EDF-2018':
        return dataset, 'EDF_2018', end_fold, end_nr, leave_out
    return None, None, None, None, None
