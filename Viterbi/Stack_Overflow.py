#Source: https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm -
import numpy as np

def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    print("T1: \n", T1, "\n \n T2: \n ", T2, "\n\n")

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1) #[[0.4], [0,3]], shape = (2,1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)
        print("\n i:", i, "\nT1: \n", T1, "\n \n T2: \n ", T2, "\n\n")

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2

#observations: normal = 0, cold = 1, dizzy = 2
#states: Healthy = 0, Fever = 1
y = np.array([0, 1, 2]) #observation state sequence; expl.: observations are normal(0) then cold (1) then dizzy (2)
A = np.array([[0.7, 0.3], [0.4, 0.6]]) #state transition matrix: e.g. prob of 0.7 to transition from Healthy to Healthy, prob of 0.3 to transition from healthy to fever
B = np.array([[0.5, 0.4, 0.1],[0.1, 0.3, 0.6]]) #Emission matrix: e.g. prob of 0.5 to feel normal if you're healthy, prob of 0.4 to feel cold when healthy
Pi = np.array([0.6, 0.4]) #initial distribution, expl.: prob. of 0.6 to start healthy, prob of 0.4 to start ill
x, T1, T2 = viterbi(y, A, B, Pi)
print(x, T1, T2)
# x: most likely trajectory: e.g. [0 0 1] means Healthy, Healthy, Fever
# T1: probability of most likely path so far, e.g.:
#   [[0.3     0.084   0.00588]
#   [0.04    0.027   0.01512]]
# means prob. of 0.3 to be in Healthy first, prob. of 0.04 to start in Fever; prob of 0.084 to have healthy - healthy;
# prob of 0.027 to have healthy - fever (only the most likely path is considered) etc.
# T2: the x_j-1 of the most likely path so far? The previous state that maximizes probability