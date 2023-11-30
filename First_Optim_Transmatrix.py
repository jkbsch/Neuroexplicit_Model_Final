# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights

import torch.nn as nn
from DNN_and_Vit import *


# Linear regression
# f = w * x

# here : f = 2 * x

class First_Optim_Transmatrix():
    def __init__(self):

        # Device configuration
        #torch.autograd.set_detect_anomaly(True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.source = DnnAndVit(print_info=False)

        self.sleepy, self.labels = self.load_data()
        self.one_hot = nn.functional.one_hot(self.labels)
        self.trans = self.trainable()

        #print(f'Prediction before training: f(5) = {self.forward(5).item():.3f}')

        # 2) Define loss and optimizer
        self.learning_rate = 0.00001
        self.n_iters = 200

        # callable function
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam([self.trans], lr=self.learning_rate)

        self.training()



    def load_data(self):
        # 0) Training samples
        """X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
        return X, Y"""
        labels = torch.from_numpy(self.source.P_Matrix_labels).to(device=self.device, dtype=torch.int64)
        sleepy = torch.from_numpy(self.source.P_Matrix_probs).to(device=self.device, dtype=torch.float64)
        return sleepy, labels

    def trainable(self):
        # 1) Design Model: Weights to optimize and forward function
        """w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        return w"""
        trans = torch.from_numpy(self.source.Transition_Matrix).to(device=self.device, dtype=torch.float64)
        trans.requires_grad_()
        return trans

    def forward(self):
        res = Viterbi(self.trans, self.sleepy, return_log=True, print_info=False)
        res_probs = torch.div(res.T1,torch.sum(res.T1, dim=0))
        return res.x, res.T1, res_probs

    def training(self):
        # 3) Training loop
        for epoch in range(self.n_iters):
            # predict = forward pass
            _, y_predicted_unnormalized, y_predicted_normalized = self.forward()

            # loss
            l = self.loss(torch.transpose(y_predicted_unnormalized, 0, 1), self.labels)

            # calculate gradients = backward pass
            l.backward()

            # update weights

            self.optimizer.step()

            # zero the gradients after updating
            self.optimizer.zero_grad()

            if epoch % 10 == 0:
                print('epoch ', epoch + 1, ': new Trans Matrix = ', self.trans, ' loss = ', l)

        #print(f'Prediction after training: f(5) = {self.forward(5).item():.3f}')



def main():
    optimized_trans_matrix = First_Optim_Transmatrix()
    out_name = "./Transition_Matrix/"
    np.savetxt(out_name + "first_optimized_trans_matrix.txt", optimized_trans_matrix.trans.detach().numpy(), fmt="%.15f", delimiter=",")


if __name__ == "__main__":
    main()
