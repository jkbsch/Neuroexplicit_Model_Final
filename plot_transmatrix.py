import numpy as np
import matplotlib.pyplot as plt
import HMM_utils


def plot_transmatrix(trans_matr="EDF_2018", oalpha=False, otrans=True, successful=False, fold=20, lr=0.01, alpha=0.3, epochs=1, checkpoints='given', FMMIE=False, mlength=10, trwtest=True, startalpha=0.1):
    fig, ax = plt.subplots()
    res_alpha, transmatrix = HMM_utils.load_Transition_Matrix(trans_matr=trans_matr, oalpha=False, otrans=False, fold=fold, check=False,
                                                    successful=successful,
                                                    checkpoints=checkpoints, lr=lr, alpha=alpha, epochs=epochs, FMMIE=FMMIE, mlength=mlength, trwtest=trwtest, startalpha=startalpha)



    ax.matshow(transmatrix, cmap='Purples')
    ax.set_yticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])
    ax.set_xticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])

    for i in range(5):
        for j in range(5):
            c = transmatrix[j, i]
            if c < 0.6:
                ax.text(i, j, str(f'{c:.2f}'), va='center', ha='center', color='black')
            else:
                ax.text(i, j, str(f'{c:.2f}'), va='center', ha='center', color='white')

    ax.set_title(f"Transition Matrix {trans_matr}:", fontsize=12)



    fig.tight_layout()
    if res_alpha is None:
        res_alpha = float('-inf')
    plt.figtext(0.01, 0.01,
                f'Transition Matrix for {trans_matr}, evaluated on fold {fold}, with lr = {lr} and final alpha = {res_alpha:.3f} on {epochs} Epochs. Alpha trained: {oalpha}, Trans trained: {otrans}, mlength = {mlength}, trwtest: {trwtest}, startalpha: {startalpha}', size=4)

    plt.show()

    fig.savefig(
        f'results/transmatr_TM{transmatrix[-1]}oa{oalpha:0}ot{otrans:0}a{alpha:.2f}{checkpoints}e{epochs}lr{lr}FMMIE{FMMIE:0}mlen{mlength}trw{trwtest:0}sa{startalpha}.png.png',
        dpi=1200)


plot_transmatrix(trans_matr="EDF_2018", oalpha=False, otrans=False, successful=True, fold=1, lr=0.01, alpha=1.0, epochs=60, checkpoints='given')
