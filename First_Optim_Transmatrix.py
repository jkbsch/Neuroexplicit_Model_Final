# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights

import torch
import torch.nn as nn
from DNN_and_Vit import *


# Linear regression
# f = w * x

# here : f = 2 * x

class First_Optim_Transmatrix():
    def __init__(self):

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.source = DnnAndVit(print_info=False)

        self.sleepy, self.labels = self.load_data()
        self.trans = self.trainable()

        print(f'Prediction before training: f(5) = {self.forward(5).item():.3f}')

        # 2) Define loss and optimizer
        self.learning_rate = 0.01
        self.n_iters = 100

        # callable function
        self.loss = nn.MSELoss()

        self.optimizer = torch.optim.SGD([self.trans], lr=self.learning_rate)

        self.training()

    def load_data(self):
        # 0) Training samples
        """X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
        return X, Y"""
        labels = torch.from_numpy(self.source.P_Matrix_labels).to(device=self.device, dtype=torch.int32)
        sleepy = torch.from_numpy(self.source.P_Matrix_probs).to(device=self.device, dtype=torch.float64)
        return sleepy, labels

    def trainable(self):
        # 1) Design Model: Weights to optimize and forward function
        """w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        return w"""
        trans = torch.from_numpy(self.source.Transition_Matrix).to(device=self.device, dtype=torch.float64)
        trans.requires_grad_()
        return trans

    def forward(self, x):
        return self.w * x

    def training(self):
        # 3) Training loop
        for epoch in range(self.n_iters):
            # predict = forward pass
            y_predicted = self.forward(self.X)

            # loss
            l = self.loss(self.Y, y_predicted)

            # calculate gradients = backward pass
            l.backward()

            # update weights
            self.optimizer.step()

            # zero the gradients after updating
            self.optimizer.zero_grad()

            if epoch % 10 == 0:
                print('epoch ', epoch + 1, ': w = ', self.w, ' loss = ', l)

        print(f'Prediction after training: f(5) = {self.forward(5).item():.3f}')


def main():
    First_Optim_Transmatrix()


if __name__ == "__main__":
    main()
