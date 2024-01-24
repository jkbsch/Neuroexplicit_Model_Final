import numpy as np
import matplotlib.pyplot as plt
import HMM_utils


def plot_transmatrix(trans_matr="EDF_2018", oalpha=False, otrans=True, successful=False, fold=20, lr=0.01, alpha=0.3,
                     epochs=1, checkpoints='given', FMMIE=False, mlength=10, trwtest=True, startalpha=0.1):
    fig, ax = plt.subplots()
    res_alpha, transmatrix = HMM_utils.load_Transition_Matrix(trans_matr=trans_matr, oalpha=False, otrans=False,
                                                              fold=fold, check=False,
                                                              successful=successful,
                                                              checkpoints=checkpoints, lr=lr, alpha=alpha,
                                                              epochs=epochs, FMMIE=FMMIE, mlength=mlength,
                                                              trwtest=trwtest, startalpha=startalpha)

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
                f'Transition Matrix for {trans_matr}, evaluated on fold {fold}, with lr = {lr} and final alpha = {res_alpha:.3f} on {epochs} Epochs. Alpha trained: {oalpha}, Trans trained: {otrans}, mlength = {mlength}, trwtest: {trwtest}, startalpha: {startalpha}',
                size=4)

    plt.show()

    fig.savefig(
        f'results/transmatr_TM{transmatrix[-1]}oa{oalpha:0}ot{otrans:0}a{alpha:.2f}{checkpoints}e{epochs}lr{lr}FMMIE{FMMIE:0}mlen{mlength}trw{trwtest:0}sa{startalpha}.png.png',
        dpi=1200)


def plot_bar_context():
    context1 = np.array([[609, 971, 1621, 407, 1437], [537, 907, 1280, 333, 1261]])
    diff1 = context1[1] - context1[0]
    context2 = np.array([[68, 448, 61, 302, 7], [68, 515, 74, 320, 6]])
    diff2 = context2[1] - context2[0]
    context3 = np.array([[1341, 2312, 2655, 1735, 583], [1419, 2344, 2677, 1807, 598]])
    diff3 = context3[1] - context3[0]

    labels = ['W', 'N1', 'N2', 'N3', 'REM', 'W ', 'N1 ', 'N2 ', 'N3 ', 'REM ', 'W  ', 'N1  ', 'N2  ', 'N3  ', 'REM  ']
    res = np.concatenate((diff1, diff2, diff3))

    """fig, ax = plt.subplots(1, 3)
    # share y axis for ax[0], ax[1] and ax[2]
    ax[0].set(ylim=(-250, 80))
    ax[0].set_title('Context 1')
    ax[1].set_title('Context 2')
    ax[2].set_title('Context 3')
    ax[1].sharey(ax[0])
    ax[2].sharey(ax[0])
    ax[0].grid(axis='y')
    ax[1].grid(axis='y')
    ax[2].grid(axis='y')
    # ax[1].get_yaxis().set_visible(False)
    # ax[2].get_yaxis().set_visible(False)




    ax[0].bar(labels, diff1, label='Context 1')
    ax[1].bar(labels, diff2, label='Context 2')
    ax[2].bar(labels, diff3, label='Context 3')"""

    fig, ax = plt.subplots()
    ax.set(ylim=(-250, 100))
    plt.grid(axis='y')
    ax.bar(labels[0:5], res[0:5], label='Constant phase', color='navy')
    ax.bar(labels[5:10], res[5:10], label='Constant phase with one outlier', color='cornflowerblue')
    ax.bar(labels[10:15], res[10:15], label='Phase with rapid changes', color='mediumslateblue')
    plt.title('Difference in Number of Errors in Context')
    plt.yticks([-350, -300, -250, -200, -150, -100, -50, 0, 50, 100], [350, 300, 250, 200, 150, 100, 50, 0, 50, 100])

    plt.legend()

    fig.tight_layout()
    plt.show()

    fig.savefig(f'results/context_v1.png', dpi=1200)



# plot_transmatrix(trans_matr="EDF_2018", oalpha=False, otrans=False, successful=True, fold=1, lr=0.01, alpha=1.0, epochs=60, checkpoints='given')
plot_bar_context()