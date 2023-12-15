import numpy as np
import matplotlib.pyplot as plt
import HMM_utils


def plot_transmatrix(trans_matr="EDF_2013", oalpha=False, otrans=True, successful=False, fold=20, lr=0.01, alpha=0.3, epochs=1, checkpoints='given'):
    fig, ax = plt.subplots(2, 1)
    res_alpha1, transmatrix1 = HMM_utils.load_Transition_Matrix(trans_matr=trans_matr, oalpha=False, otrans=False, fold=fold, check=False,
                                                    successful=successful,
                                                    checkpoints=checkpoints, lr=lr, alpha=alpha, epochs=epochs)

    res_alpha2, transmatrix2 = HMM_utils.load_Transition_Matrix(trans_matr=trans_matr, oalpha=oalpha, otrans=otrans, fold=fold, check=False,
                                                    successful=successful,
                                                    checkpoints=checkpoints, lr=lr, alpha=alpha, epochs=epochs)

    ax[0].matshow(transmatrix1, cmap='magma')
    ax[0].set_yticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])
    ax[0].set_xticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])

    for i in range(5):
        for j in range(5):
            c = transmatrix1[j, i]
            if c >= 0.6:
                ax[0].text(i, j, str(f'{c:.2f}'), va='center', ha='center', color='black')
            else:
                ax[0].text(i, j, str(f'{c:.2f}'), va='center', ha='center', color='white')

    ax[0].set_title("Not-optimized Transition Matrix of Dataset Sleep-EDF-2013:", fontsize=12)

    ax[1].matshow(transmatrix2, cmap='magma')
    ax[1].set_yticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])
    ax[1].set_xticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])

    for i in range(5):
        for j in range(5):
            c = transmatrix2[j, i]
            if c >= 0.6:
                ax[1].text(i, j, str(f'{c:.2f}'), va='center', ha='center', color='black')
            else:
                ax[1].text(i, j, str(f'{c:.2f}'), va='center', ha='center', color='white')

    ax[1].set_title("Optimized Transition Matrix of Dataset Sleep-EDF-2013:", fontsize=12)

    fig.tight_layout()

    plt.figtext(0.01, 0.01,
                f'Transition Matrices for {trans_matr}, evaluated on fold {fold}, with lr = {lr} and final alpha = {res_alpha2:.3f} on {epochs} Epochs. Alpha trained: {oalpha}, Trans trained: {otrans}', size=6)

    plt.show()

    fig.savefig(
        f'results/transmatr_{trans_matr}_fold_{fold}_lr_{lr}_alpha_{res_alpha2}_epochs_{epochs}_oalpha_{oalpha}_otrans_{otrans}.png',
        dpi=1200)


plot_transmatrix(trans_matr="EDF_2013", oalpha=False, otrans=True, successful=False, fold=4, lr=0.0001, alpha=0.3, epochs=60, checkpoints='given')
