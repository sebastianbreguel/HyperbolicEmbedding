import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input, 1)

    def forward(self, x):
        x = self.l1(x)
        return x

    def predict(self, test_x):
        x = self.l1(test_x)
        return x
