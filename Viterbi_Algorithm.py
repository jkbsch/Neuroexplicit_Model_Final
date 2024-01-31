# Source: https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm checked 05th Oct
# 2023 - adapted
import numpy as np
import torch


class Viterbi:
    """
            Applies the Viterbi algorithm to a given HMM model, and returns the most likely state trajectory.
            It is also able to calculate the k-best paths, and the MMIE loss.
            It works with numpy arrays and torch tensors.
    """

    def __init__(self, A, P, Pi=None, logscale=True, alpha=None, return_log=None, print_info=True, softmax=False,
                 FMMIE=False, labels=None, k_best=1):
        """
        :param A: Transition matrix of the HMM model
        :param P: P-matrix of the HMM model
        :param Pi: Initial distribution of the HMM model; if not given, it is set to uniform
        :param logscale: Whether the calculation is done in logscale; Caution: most calculations can only be realized in
                        logscale, such as the calculation with alpha
        :param alpha: Parameter alpha for weighing the transition matrix and the P-matrix
        :param return_log: Whether the output should be in logscale
        :param print_info: Whether additional information should be printed
        :param softmax: Whether softmax is used in the Viterbi algorithm; Does not work for FMMIE or k_best > 1
        :param FMMIE: Whether the FMMIE loss should be calculated
        :param labels: Labels of the dataset; only needed for FMMIE
        :param k_best: Number of the best paths returned by the Viterbi algorithm

        Find more information of how the (k-best) Viterbi algorithm works in the corresponding bachelor's thesis
        """
        # check if the input is a torch tensor or a numpy array
        if torch.is_tensor(A):
            self.is_torch = True
            self.device = A.device
        else:
            self.is_torch = False

        # Initialize the model given the parameters
        self.A = A
        self.P = P
        self.logscale = logscale
        self.print_info = print_info
        self.softmax = softmax
        self.labels = labels
        self.k_best = k_best

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

        if self.k_best > 1:
            if self.logscale is False or self.is_torch is False or self.softmax is True:
                raise NotImplementedError
        elif self.softmax and not self.is_torch:
            raise NotImplementedError

        self.T = len(P)  # length of the sequence
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

        if FMMIE:
            self.x, self.res_FMMIE = self.calc_FMMIE()  # calculate the FMMIE loss
        elif self.is_torch:
            if k_best > 1:  #  k-best paths
                self.x, self.T1, self.T2, self.T3, self.y = self.calc_viterbi_k_best()
            else:  # 1-best path
                self.x, self.T1, self.T2, self.T3, self.y = self.calc_viterbi()
        else:  # numpy
            self.x, self.T1, self.T2 = self.calc_viterbi()

    # check if alpha is None and set it to 1 if it is
    def alpha(self, alpha):
        if alpha is None:
            if self.is_torch:
                return torch.tensor([1.0], dtype=torch.float64, device=self.device)
            else:
                return 1.0
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
            if self.softmax:
                T3 = torch.zeros((self.K, self.T), dtype=torch.float64, device=self.device)
        else:
            T1 = np.empty((self.K, self.T), 'd')
            T2 = np.empty((self.K, self.T), 'B')

        if self.logscale:
            T1[:, 0] = self.Pi + self.P[0]

        else:
            T1[:, 0] = self.Pi * self.P[0]

        T2[:, 0] = 0

        # Iterate through the observations updating the tracking tables
        for i in range(1, self.T):
            if self.logscale:
                if self.is_torch:
                    T1[:, i] = torch.max(T1[:, i - 1] + self.alpha * self.A.T, 1).values + self.P[i]
                    T2[:, i] = torch.argmax(T1[:, i - 1] + self.alpha * self.A.T, 1)
                    if self.softmax:
                        temp = torch.nn.functional.softmax(T1[:, i - 1] + self.alpha * self.A.T, 1)
                        T3[:, i] = torch.matmul(temp, torch.arange(self.K, dtype=torch.float64)[:, None]).squeeze(1)

                else:
                    T1[:, i] = np.max(T1[:, i - 1] + self.alpha * self.A.T, 1) + self.P[i]
                    T2[:, i] = np.argmax(T1[:, i - 1] + self.alpha * self.A.T, 1)

            else:
                if self.is_torch:
                    T1[:, i] = torch.max(T1[:, i - 1] * self.A.T, 1).values * self.P[i]
                    T2[:, i] = torch.argmax(T1[:, i - 1] * self.A.T, 1)
                else:
                    T1[:, i] = np.max(T1[:, i - 1] * self.A.T, 1) * self.P[i]

                    T2[:, i] = np.argmax(T1[:, i - 1] * self.A.T, 1)

        # Build the output, optimal model trajectory
        if self.is_torch:
            x = torch.empty(self.T, dtype=torch.int64, device=self.device)

            x[-1] = torch.argmax(T1[:, self.T - 1])

            if self.softmax:
                y = torch.empty(self.T, dtype=torch.float64, device=self.device)
                temp = torch.nn.functional.softmax(T1[:, self.T - 1], dim=0)
                y[-1] = torch.matmul(temp, torch.arange(self.K, dtype=torch.float64)[:, None])
            else:
                T3, y = None, None
        else:
            x = np.empty(self.T, 'B')
            x[-1] = np.argmax(T1[:, self.T - 1])

        for i in reversed(range(1, self.T)):
            x[i - 1] = T2[x[i], i]
            if self.is_torch and self.softmax:
                y[i - 1] = T3[x[i], i]

        if self.logscale != self.return_log:  # convert to logscale if necessary
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

    def calc_viterbi_k_best(self):
        # Initialize the tracking tables
        T1 = torch.full((self.k_best, self.K, self.T), fill_value=float('-inf'), dtype=torch.float64,device=self.device)
        T2 = torch.full((self.k_best, self.K, self.T, 2), fill_value=-1, dtype=torch.int, device=self.device)

        # only the first-best path can be given for initialization; It is initialized with the initial distribution
        T1[0, :, 0] = self.Pi + self.P[0]
        T2[0, :, 0] = 0

        # Iterate through the observations updating the tracking tables
        for i in range(1, self.T):
            temp = T1[:, :, i - 1] + self.alpha * self.A.T[:, None]
            flattened_vals, flattened_idx = torch.topk(temp.flatten(start_dim=1, end_dim=2), self.k_best)
            T1[:, :, i] = flattened_vals.T + self.P[i]
            unraveled = np.array(np.unravel_index(flattened_idx.numpy(), shape=(self.k_best, self.K))).T
            T2[:, :, i, :] = torch.tensor(unraveled)


        final_vals, final_idx = torch.topk((T1[:, :, -1]).flatten(), self.k_best)
        last_valid = torch.where(final_vals == float('-inf'))[0]
        if len(last_valid) > 0:  # check if there are less than k_best valid paths
            last_valid = last_valid[0]
            final_idx = final_idx[0:last_valid]
            final_vals = final_vals[0:last_valid]
            x = torch.empty((last_valid, self.T), dtype=torch.int64, device=self.device)
        else:
            x = torch.empty((self.k_best, self.T), dtype=torch.int64, device=self.device)

        final_state = final_idx % self.K  # find the last state of the k-best paths
        k_prev = final_idx // self.K
        x[:, -1] = final_state

        # iterate through T to find the previous states
        for i in reversed(range(1, self.T)):
            x[:,i-1] = T2[k_prev, x[:,i], i, 1]
            k_prev = T2[k_prev, x[:,i], i, 0]

        return x, T1, T2, None, None

    def calc_FMMIE(self):
        if not self.logscale:
            raise NotImplementedError
        self.k_best += 1 # if the correct path is in one of the k-best paths, then we need to have one more path

        if not self.is_torch:  #  finding k best paths only works with torch tensors
            self.A = torch.from_numpy(self.A)
            self.P = torch.from_numpy(self.P)
            if self.Pi is not None:
                self.Pi = torch.from_numpy(self.Pi)
        else:
            A_optimizable = torch.clone(self.A)  # finding k_best paths should not be used for backpropagation
            if torch.is_tensor(self.alpha):
                alpha_optimizable = torch.clone(self.alpha)
                self.alpha.detach()
            else:
                alpha_optimizable = self.alpha
            self.A.detach()
        best_paths, _, _, _, _ = self.calc_viterbi_k_best()  # only the k_best paths are of interest

        if not self.is_torch:  # numpy
            best_paths = best_paths.numpy()
            exclude_path = len(best_paths)  # exclude the path that equals the correct one, or exclude the last path
            for i in range(len(best_paths) - 1):
                if np.all(self.labels, best_paths[i]):
                    exclude_path = i
                    break

            # numerator numpy:
            num = self.Pi[self.labels[0]] + np.sum(self.P[np.arange(len(self.labels)), self.labels])
            transitions = np.zeros((self.K, self.K))
            for i in range(len(self.labels) - 1):
                transitions[self.labels[i], self.labels[i + 1]] += 1
            num += self.alpha * (transitions * self.A).sum()
            num = np.exp(num)

            # denumerator numpy:
            den = 0
            transitions = np.zeros((self.K, self.K))

            for i in range(len(best_paths)):
                if i == exclude_path:
                    continue
                den_temp = self.Pi[best_paths[i][0]] + np.sum(self.P[np.arange(len(best_paths[i])), best_paths[i]])
                transitions = transitions * 0
                for j in range(len(best_paths[i]) - 1):
                    transitions[best_paths[i][j], best_paths[i][j + 1]] += 1
                den_temp += self.alpha * (transitions * self.A).sum()
                den += np.exp(den_temp)
            den = np.log(den)

        else:  # torch
            exclude_path = len(best_paths) # exclude the path that equals the correct one, or exclude the last path
            for i in range(len(best_paths) -1):
                if torch.equal(self.labels, best_paths[i]):
                    exclude_path = i
                    break

            # numerator torch
            try:
                num = self.Pi[self.labels[0]] + torch.sum(self.P[torch.arange(len(self.labels)), self.labels])
            except:
                print('error')
            transitions = torch.zeros((self.K, self.K))
            for i in range(len(self.labels) - 1):
                transitions[self.labels[i], self.labels[i + 1]] += 1
            num = num + alpha_optimizable * (transitions * A_optimizable).sum()

            # denumerator torch
            den = 0
            transitions = torch.zeros((self.K, self.K))

            for i in range(len(best_paths)):
                if i == exclude_path:
                    continue
                den_temp = self.Pi[best_paths[i][0]] + torch.sum(self.P[torch.arange(len(best_paths[i])), best_paths[i]])
                transitions = transitions * 0
                for j in range(len(best_paths[i]) - 1):
                    transitions[best_paths[i][j], best_paths[i][j + 1]] += 1
                den_temp = den_temp + alpha_optimizable * (transitions * A_optimizable).sum()
                den = den + den_temp.exp()
            den = torch.log(den)



        return best_paths[0], -(num - den)


def main():

    # here are some exemplary inputs for the Viterbi algorithm
    A = np.array([[0.7, 0.3], [0.9, 0.1]])
    Pi = np.array([0.8, 0.2])
    P = np.array([[0.1, .9], [0.9, 0.1], [0.01, 0.99]])
    labels = np.array([1, 0, 1])
    alpha = 1

    AT = torch.from_numpy(A).to(dtype=torch.float64)
    AT.requires_grad = True
    PiT = torch.from_numpy(Pi).to(dtype=torch.float64)
    PT = torch.from_numpy(P).to(dtype=torch.float64)
    labelsT = torch.from_numpy(labels).to(dtype=torch.int64)
    alphaT = torch.tensor([1], dtype=torch.float64, requires_grad=True)



    A1 = torch.from_numpy(A).to(dtype=torch.float64, device='cpu')
    Pi1 = torch.from_numpy(Pi).to(dtype=torch.float64, device='cpu')
    P1 = torch.from_numpy(P).to(dtype=torch.float64, device='cpu')


    Viterbi_2 = Viterbi(A1, P1, Pi1, logscale=True, return_log=True, k_best=3, FMMIE=True, labels = labelsT)
    Viterbi_1 = Viterbi(A, P, Pi, logscale=True, return_log=True, alpha=1, k_best=1)
    x_1, T1_1, T2_1 = Viterbi_1.x, Viterbi_1.T1, Viterbi_1.T2
    x_2, T1_2, T2_2 = Viterbi_2.x.numpy(), Viterbi_2.T1.numpy(), Viterbi_2.T2.numpy()

    print(x_1 == x_2, np.round(T1_1, 4) == np.round(T1_2, 4), T2_1 == T2_2)
    print(T2_1)
    print(T2_2)


if __name__ == "__main__":
    main()
