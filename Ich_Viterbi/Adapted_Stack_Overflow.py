#Source: https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm checked 05th Oct 2023
import numpy as np

def viterbi(A, P, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details. Example: [[1->1, 1->2],[2->1, 2->2]]
    P: array(T, K)
        Probability of being of state k given observation in time t - calculated via DNN.
        Example: [[Prob. of 0.3 to be in state 1 at time 0 given the data, Prob of 0.7 to be in state 2 at t = 0], [prob of 0.4 for state 1 at time 1, prob of 0.6 for state 2 at time 2]]
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).



    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
        Example: [0 0 1] means that the most likely a posteriori path is state 0, state 0, state 1
    T1: array (K, T)
        the probability of the most likely path so far
        Example:
        [[0.3     0.084   0.00588]
       [0.04    0.027   0.01512]]
       means prob. of 0.3 to be in State 0 at time 0 and  prob. of 0.04 to be in Staate 1 at t=0;  prob. of 0.084 to be in State 0 at t=1, (c.f. T2 to find the most likely path to get there);
       prob of 0.027 to be in State 1 at time t=1 (c.f. T2 to find the most likely path to get there)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(P)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initialize the tracking tables from first observation
    #T1[:, 0] = Pi * B[:, y[0]]
    #print("B[:, y[0]]: \n", B[:, y[0]], "\n")

    T1[:, 0] = Pi * P[0]

    T2[:, 0] = 0
    print("T1: \n", T1,"\n \n T2: \n ", T2, "\n\n")

    # Iterate through the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * (P[np.newaxis, 1]).T, 1) # multipliziere Wkt des letzten States mit Transitionswahrscheinlichkeit mit Wkt für den aktuellen Zustand aus DNN; suche den State aus vorheriger Periode, der Wkt maximiert
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)
        print("\n i:",i, "\nT1: \n", T1,"\n \n T2: \n ", T2, "\n\n")

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2

#observations: normal = 0, cold = 1, dizzy = 2
#states: Healthy = 0, Fever = 1
#y = np.array([0, 1, 2]) #observation state sequence; expl.: observations are normal(0) then cold (1) then dizzy (2)
A = np.array([[0.7, 0.3], [0.4, 0.6]]) #state transition matrix: e.g. prob of 0.7 to transition from Healthy to Healthy, prob of 0.3 to transition from healthy to fever
#B = np.array([[0.5, 0.4, 0.1],[0.1, 0.3, 0.6]]) #Emission matrix: e.g. prob of 0.5 to feel normal if you're healthy, prob of 0.4 to feel cold when healthy
Pi = np.array([0.6, 0.4]) #initial distribution, expl.: prob. of 0.6 to start healthy, prob of 0.4 to start ill
P = np.array([[0.3, .7], [0.7, 0.3], [0.5, 0.5]]) #e.g. DNN has calculated that - given the data - there's a 30% chance that the first state is Healthy (shape 3,2)


x, T1, T2 = viterbi(A, P, Pi)
print(x, T1, T2)


# x: most likely trajectory: e.g. [0 0 1] means Healthy, Healthy, Fever
# T1: probability of most likely path so far, e.g.:
#   [[0.3     0.084   0.00588]
#   [0.04    0.027   0.01512]]
# means prob. of 0.3 to be in Healthy first, prob. of 0.04 to start in Fever; prob of 0.084 to have healthy - healthy;
# prob of 0.027 to have healthy - fever (only the most likely path is considered) etc.
# T2: the x_j-1 of the most likely path so far?