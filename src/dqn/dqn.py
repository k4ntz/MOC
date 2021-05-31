# nn for dq learning

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(9216, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, outputs)

    # Called with one element to determine next action
    def forward(self, x):
        x = x.to(device)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x