import torch
import torch.nn as nn


class AFDNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AFDNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def flatten(self, fea):
        return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))

    def forward(self, x):
        out =  self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out.squeeze()