import numpy as np
import matplotlib.pyplot as plt
import HMM_utils


def plot_transmatrix():
    fig, ax = plt.subplots(2, 1)
    transmatrix1 = HMM_utils.load_Transition_Matrix(trans_matr="EDF_2013", optimized=False, fold=1, check=False, successful=True,
                           checkpoints='given', lr=0.001, alpha=0.3, epochs=60)

    transmatrix2 = HMM_utils.load_Transition_Matrix(trans_matr="EDF_2013", optimized=True, fold=1, check=False, successful=False,
                           checkpoints='given', lr=0.001, alpha=0.3, epochs=60)

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

    plt.show()

plot_transmatrix()