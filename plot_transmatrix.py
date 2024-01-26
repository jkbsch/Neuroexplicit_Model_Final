import numpy as np
import matplotlib.pyplot as plt
import HMM_utils
import Optimize_alpha
import matplotlib as mpl
import HMM_utils


def plot_transmatrix(trans_matr="EDF_2018", oalpha=False, otrans=True, successful=False, fold=20, lr=0.01, alpha=0.3,
                     epochs=1, checkpoints='given', FMMIE=False, mlength=10, trwtest=True, startalpha=0.1):
    fig, ax = plt.subplots()

    res_alpha, transmatrix = HMM_utils.load_Transition_Matrix(trans_matr=trans_matr, oalpha=oalpha, otrans=otrans,
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



    fig.savefig(
        f'results_n/transmatr_TM{transmatrix[-1]}oa{oalpha:0}ot{otrans:0}a{alpha:.2f}{checkpoints}e{epochs}lr{lr}FMMIE{FMMIE:0}mlen{mlength}trw{trwtest:0}sa{startalpha}.png.png',
        dpi=1200)

    plt.show()


def plot_bar_context():
    context1 = np.array([[609, 971, 1621, 407, 1437], [537, 907, 1280, 333, 1261]])
    diff1 = context1[1] - context1[0]
    context2 = np.array([[68, 448, 61, 302, 7], [68, 515, 74, 320, 6]])
    diff2 = context2[1] - context2[0]
    context3 = np.array([[1341, 2312, 2655, 1735, 583], [1419, 2344, 2677, 1807, 598]])
    diff3 = context3[1] - context3[0]



    labels = ['W', 'N1', 'N2', 'N3', 'REM', 'W', 'N1', 'N2', 'N3', 'REM', 'W', 'N1', 'N2', 'N3', 'REM']
    res = np.concatenate((diff1, diff2, diff3))

    fig, ax = plt.subplots(1, 3)
    ax[0].set(ylim=(-250, 100))
    ax[1].sharey(ax[0])
    ax[2].sharey(ax[0])
    ax[0].grid(axis='y')
    ax[1].grid(axis='y')
    ax[2].grid(axis='y')
    ax[0].bar(labels[0:5], res[0:5], label='Constant phase', color='mediumslateblue') # navy
    ax[1].bar(labels[5:10], res[5:10], label='Constant phase with one outlier', color='mediumslateblue') # cornflowerblue
    ax[2].bar(labels[10:15], res[10:15], label='Phase with rapid changes', color='mediumslateblue') # mediumslateblue
    # plt.title('Difference in Number of Errors in Context')
    fontsize=10
    ax[0].set_title('Constant phase', fontsize=fontsize)
    ax[1].set_title('Constant phase with one outlier', fontsize=fontsize)
    ax[2].set_title('Phase with rapid changes', fontsize=fontsize)
    plt.yticks([-350, -300, -250, -200, -150, -100, -50, 0, 50, 100], [350, 300, 250, 200, 150, 100, 50, 0, 50, 100])

    # plt.legend()

    fig.tight_layout()
    plt.show()

    fig.savefig(f'results_n/context_v1.png', dpi=1200)

def plot_difference_confusion():
    alpha = 1.0
    otrans = True
    oalpha = False
    lr = 0.00001
    epochs = 100
    checkpoints = 'given'
    max_length = 10
    mlength = 1000
    startalpha = 1.0
    used_set='val'
    eval1 = Optimize_alpha.OptimizeAlpha(used_set=used_set, dataset='Sleep-EDF-2018', start_alpha=alpha, end_alpha=alpha, step=0.05,
                                   print_all_results=False, trans_matrix=None, otrans=otrans, oalpha=oalpha,
                                   evaluate_result=True, visualize=False,
                                   optimize_alpha=False, lr=lr, successful=True, epochs=epochs, checkpoints=checkpoints,
                                   max_length=max_length, FMMIE=True, mlength=mlength, trwtest=True, startalpha=startalpha)
    confusion_matrix_1 = eval1.res[-1]

    alpha = 0.0
    max_length = None
    eval2 = Optimize_alpha.OptimizeAlpha(used_set=used_set, dataset='Sleep-EDF-2018', start_alpha=alpha, end_alpha=alpha,
                                         step=0.05,
                                         print_all_results=False, trans_matrix=None, otrans=otrans, oalpha=oalpha,
                                         evaluate_result=True, visualize=False,
                                         optimize_alpha=False, lr=lr, successful=True, epochs=epochs,
                                         checkpoints=checkpoints,
                                         max_length=max_length, FMMIE=True, mlength=mlength, trwtest=True,
                                         startalpha=startalpha)
    confusion_matrix_2 = eval2.res[-1]
    difference = confusion_matrix_1 - confusion_matrix_2

    arr = np.ones(np.shape(difference))
    np.fill_diagonal(arr, -1)

    norm = mpl.colors.Normalize(vmin=-0.1, vmax=0.1)

    fig, ax = plt.subplots()
    ax.matshow(difference*arr, cmap='bwr', norm=norm)
    ax.set_yticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])
    ax.set_xticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])

    for i in range(5):
        for j in range(5):
            c = difference[j, i] * 100
            if c < 60:
                ax.text(i, j, str(f'{c:.1f}%'), va='center', ha='center', color='black')
            else:
                ax.text(i, j, str(f'{c:.1f}%'), va='center', ha='center', color='cornsilk')
    plt.title('Predicted Class', fontsize=10)
    # plt.xaxis.set_label_position('top')
    # set x label to the top of the plot

    plt.ylabel('Actual Class')

    fig.tight_layout()

    plt.savefig(f'results_n/difference_confusion_matrix.png', dpi=1200)
    plt.show()

def plot_difference_transition(average = False):
    trans_matrix = "EDF_2018"
    oalpha = True
    otrans = True
    fold = 1

    lr = 0.001
    epochs = 100
    mlength = 10
    startalpha = 1.0




    alpha1, trans_matr_1 = HMM_utils.load_Transition_Matrix(trans_matrix, checkpoints='given', oalpha=oalpha, otrans=otrans, fold=fold, lr=lr, successful=True, epochs=epochs, FMMIE=True, mlength=mlength, trwtest=True, startalpha=startalpha)

    if not average:

        alpha2, trans_matr_2 = HMM_utils.load_Transition_Matrix(trans_matrix, checkpoints='given', oalpha=oalpha, otrans=otrans, fold=fold, lr=lr, successful=True, epochs=epochs, FMMIE=True, mlength=mlength, trwtest=True, startalpha=startalpha)
        difference = trans_matr_1 - trans_matr_2
    else:
        difference = np.zeros(np.shape(trans_matr_1))
        all_transmatr = [trans_matr_1]
        all_alphas = [alpha1]
        for fold in range(2,11):
            alpha, trans_matr = HMM_utils.load_Transition_Matrix(trans_matrix, checkpoints='given', oalpha=oalpha, otrans=otrans,
                                                                    fold=fold, lr=lr, successful=True, epochs=epochs,
                                                                    FMMIE=True, mlength=mlength, trwtest=True,
                                                                    startalpha=startalpha)
            all_transmatr.append(trans_matr)
            all_alphas.append(alpha)

        avg_transmatr = np.mean(all_transmatr, axis=0)
        for i in range(10):
            difference += np.absolute(all_transmatr[i] - avg_transmatr)

        difference = difference/10
        mean_alpha = np.mean(np.array(all_alphas))
        print(f'Mean alpha: {mean_alpha:.3f}')




    fig, ax = plt.subplots()

    if average:
        norm = mpl.colors.Normalize(vmin=0, vmax=0.2)
        ax.matshow(difference, cmap='Blues', norm=norm)
    else:
        norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)
        ax.matshow(difference, cmap='PRGn', norm=norm)
    ax.set_yticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])
    ax.set_xticks([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"])

    for i in range(5):
        for j in range(5):
            c = difference[j, i]
            if c < 0.6:
                ax.text(i, j, str(f'{c:.2f}'), va='center', ha='center', color='black')
            else:
                ax.text(i, j, str(f'{c:.2f}'), va='center', ha='center', color='white')


    fig.tight_layout()

    plt.savefig(f'results_n/difference_transition_matrix.png', dpi=1200)
    plt.show()

    if oalpha and average:

        max_alpha = np.max(np.array(all_alphas))
        if max_alpha < 0.3:
            max_alpha = 0.3
        else:
            max_alpha = int(max_alpha)+1
        fig, ax = plt.subplots()
        ax.set(ylim=(0, max_alpha))
        plt.xlabel('Fold')
        plt.ylabel('Alpha')
        ax.scatter(np.arange(1,11), all_alphas, color='cornflowerblue', marker='x')

        fig.tight_layout()

        plt.savefig(f'results_n/alphas_in_folds.png', dpi=1200)
        plt.show()

# plot_transmatrix(trans_matr="EDF_2018", oalpha=True, otrans=True, successful=True, fold=1, lr=0.001, epochs=100, checkpoints='given', FMMIE=True, mlength=10, trwtest=True, startalpha=1.0)
plot_bar_context()
# plot_difference_confusion()
# plot_difference_transition(average=True)