# Source: https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm checked 05th Oct
# 2023 - adapted
import numpy as np
import torch


class Viterbi:
    """
            Return the MAP estimate of state trajectory of Hidden Markov Model.
            Creates an object with containing the best bath.

            Inputs: ---------- A : array (K, K) State transition matrix. See HiddenMarkovModel.state_transition  for
            details. Example: [[1->1, 1->2],[2->1, 2->2]] P: array(T, K) Probability of being of state k given
            observation in time t - calculated via DNN. Example: [[Prob. of 0.3 to be in state 1 at time 0 given the
            data, Prob of 0.7 to be in state 2 at t = 0], [prob of 0.4 for state 1 at time 1, prob of 0.6 for state 2
            at time 2]] Pi: optional, (K,) Initial state probabilities: Pi[i] is the probability x[0] == i. If None,
            uniform initial distribution is assumed (Pi[:] == 1/K). logscale: optional, (bool) Defines whether the
            calculation is logarithmic or not. Default is False; if true, T1 will contain logarithmic probabilities



            Calculates: ------- x : array (T,) Maximum a posteriori probability estimate of hidden state trajectory,
            conditioned on observation sequence y under the model parameters A, B, Pi. Example: [0 0 1] means that
            the most likely a posteriori path is state 0, state 0, state 1 T1: array (K, T) the probability of the
            most likely path so far Example: [[0.3     0.084   0.00588] [0.04    0.027   0.01512]] means prob. of 0.3
            to be in State 0 at time 0 and  prob. of 0.04 to be in Staate 1 at t=0;  prob. of 0.084 to be in State 0
            at t=1, (c.f. T2 to find the most likely path to get there); prob of 0.027 to be in State 1 at time t=1 (
            c.f. T2 to find the most likely path to get there) the x_j-1 of the most likely path so far
    """


    def __init__(self, A, P, Pi=None, logscale=True, alpha=None, return_log=None, print_info=True):
        # check if the input is a torch tensor or a numpy array
        if torch.is_tensor(A):
            self.is_torch = True
            self.device = A.device
        else:
            self.is_torch = False

        # Initialize the model given the parameters
        self.A = A
        #self.A[A<0] = 0
        self.P = P
        self.logscale = logscale
        self.print_info = print_info

        # Initialize the return_log variable
        if return_log is None:
            self.return_log = self.logscale
        else:
            self.return_log = return_log

        # Cardinality of the state space
        self.K = A.shape[0]
        # Initialize the priors with default (uniform dist) if not given by caller
        if self.is_torch:
            self.Pi = Pi if Pi is not None else torch.full(size=(self.K,), fill_value=1 / self.K, device=self.device)
        else:
            self.Pi = Pi if Pi is not None else np.full(self.K, 1 / self.K)


        self.alpha = self.alpha(alpha)

        self.T = len(P)
        # convert to logscale if needed
        if self.logscale:  # convert to logscale
            if self.is_torch:
                self.A = torch.log(self.A)
                self.P = torch.log(self.P)
                self.Pi = torch.log(self.Pi)
            else:
                self.A = np.log(self.A)
                self.P = np.log(self.P)
                self.Pi = np.log(self.Pi)

        if self.is_torch:
            self.x, self.T1, self.T2, self.T3, self.y = self.calc_viterbi()
        else:
            (self.x, self.T1, self.T2) = self.calc_viterbi()

    # check if alpha is None and set it to 0.5 if it is
    def alpha(self, alpha):
        if alpha is None:
            if self.is_torch:
                return torch.tensor([0.5], dtype=torch.float64, device=self.device)
            else:
                return 0.5
        elif self.logscale is False:
            if self.print_info:
                print("[INFO]: Specifying alpha with logscale = False is not implemented. Logscale is set to True.")
            self.logscale = True
            return alpha
        else:
            return alpha

    # Calculate the most likely state trajectory using the Viterbi algorithm
    def calc_viterbi(self):

        # Initialize the tracking tables
        if self.is_torch:
            T1 = torch.empty((self.K, self.T), dtype=torch.float64, device=self.device)
            T2 = torch.empty((self.K, self.T), dtype=torch.int, device=self.device)
            T3 = torch.zeros((self.K, self.T), dtype=torch.float64, device=self.device)
        else:
            T1 = np.empty((self.K, self.T), 'd')
            T2 = np.empty((self.K, self.T), 'B')

        # Initialize the tracking tables from first observation
        # T1[:, 0] = Pi * B[:, y[0]]
        # print("B[:, y[0]]: \n", B[:, y[0]], "\n")

        if self.logscale:
            T1[:, 0] = self.Pi + self.P[0]

        else:
            T1[:, 0] = self.Pi * self.P[0]

        T2[:, 0] = 0

        # Iterate through the observations updating the tracking tables
        for i in range(1, self.T):
            if self.logscale:
                if self.is_torch:
                    T1[:, i] = torch.max(
                        T1[:, i - 1] + 2 * self.alpha * self.A.T + 2 * (1 - self.alpha) * (self.P[None, i]).T, 1).values
                    T2[:, i] = torch.argmax(T1[:, i - 1] + 2 * self.alpha * self.A.T, 1)
                    temp = torch.nn.functional.softmax(T1[:, i - 1] + 2 * self.alpha * self.A.T, 1)
                    T3[:, i] = torch.matmul(temp, torch.arange(self.K, dtype=torch.float64)[:,None]).squeeze(1)

                else:
                    T1[:, i] = np.max(
                        T1[:, i - 1] + 2 * self.alpha * self.A.T + 2 * (1 - self.alpha) * (self.P[np.newaxis, i]).T, 1)
                    # Add the probability
                    # (logscale) of the last state's occurrence to the transition probability and to the probability for
                    # the current state from the DNN. Find the state from the previous period that maximizes this
                    # probability.
                    T2[:, i] = np.argmax(T1[:, i - 1] + 2 * self.alpha * self.A.T, 1)

            else:
                if self.is_torch:
                    T1[:, i] = torch.max(T1[:, i - 1] * self.A.T * (self.P[None, i]).T, 1).values
                    T2[:, i] = torch.argmax(T1[:, i - 1] * self.A.T, 1)
                else:
                    T1[:, i] = np.max(T1[:, i - 1] * self.A.T * (self.P[np.newaxis, i]).T, 1)
                    # Multiply the probability of the last state's occurrence
                    # with the transition probability and with the probability for the current state from the DNN.
                    # Find the state from the previous period that maximizes this probability.

                    T2[:, i] = np.argmax(T1[:, i - 1] * self.A.T, 1)
            # print("\n i:",i, "\nT1: \n", T1,"\n \n T2: \n ", T2, "\n\n")

        # Build the output, optimal model trajectory
        if self.is_torch:
            x = torch.empty(self.T, dtype=torch.int64, device=self.device)
            y = torch.empty(self.T, dtype=torch.float64, device=self.device)
            # x = torch.empty(self.T, dtype=torch.float64, device=self.device)
            x[-1] = torch.argmax(T1[:, self.T - 1])
            temp = torch.nn.functional.softmax(T1[:, self.T - 1], dim=0)
            y[-1] = torch.matmul(temp, torch.arange(self.K, dtype=torch.float64)[:,None])
        else:
            x = np.empty(self.T, 'B')
            x[-1] = np.argmax(T1[:, self.T - 1])

        for i in reversed(range(1, self.T)):
            x[i - 1] = T2[x[i], i]
            if self.is_torch:
                y[i - 1] = T3[x[i], i]

        if self.logscale != self.return_log:
            if self.return_log:
                if self.is_torch:
                    T1 = torch.log(T1)
                else:
                    T1 = np.log(T1)
            else:
                if self.is_torch:
                    T1 = torch.exp(T1)
                else:
                    T1 = np.exp(T1)

        if self.is_torch:
            return x, T1, T2, T3, y
        else:
            return x, T1, T2

    """#observations: normal = 0, cold = 1, dizzy = 2 #states: Healthy = 0, Fever = 1 #y = np.array([0, 1, 
    2]) #observation state sequence; expl.: observations are normal(0) then cold (1) then dizzy (2) A = np.array([[
    0.7, 0.3], [0.9, 0.1]]) #state transition matrix: e.g. prob of 0.7 to transition from Healthy to Healthy, 
    # prob of 0.3 to transition from healthy to fever #B = np.array([[0.5, 0.4, 0.1],[0.1, 0.3, 0.6]]) #Emission 
    matrix: e.g. prob of 0.5 to feel normal if you're healthy, # prob of 0.4 to feel cold when healthy Pi = np.array(
    [0.8, 0.2]) #initial distribution, expl.: prob. of 0.6 to start healthy, prob of 0.4 to start ill P = np.array([[
    .5, .5], [0.2, 0.8], [0.4, 0.6]]) #e.g. DNN has calculated that - given the data - there's a 30% chance # that 
    the first state is Healthy (shape 3,2)"""


def main():
    A = np.array([[0.7, 0.3], [0.9, 0.1]])
    Pi = np.array([0.8, 0.2])
    P = np.array([[0.1, .9], [0.9, 0.1], [0.01, 0.99]])

    A1 = torch.from_numpy(A).to(dtype=torch.float64)
    Pi1 = torch.from_numpy(Pi).to(dtype=torch.float64)
    P1 = torch.from_numpy(P).to(dtype=torch.float64)

    Viterbi_1 = Viterbi(A, P, Pi, logscale=True, return_log=True)
    Viterbi_2 = Viterbi(A1, P1, Pi1, logscale=True, return_log=True)

    x_1, T1_1, T2_1 = Viterbi_1.x, Viterbi_1.T1, Viterbi_1.T2
    x_2, T1_2, T2_2 = Viterbi_2.x.numpy(), Viterbi_2.T1.numpy(), Viterbi_2.T2.numpy()

    print(x_1 == x_2, np.round(T1_1, 4) == np.round(T1_2, 4), T2_1 == T2_2)


if __name__ == "__main__":
    main()
