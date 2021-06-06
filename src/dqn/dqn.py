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
        # x consecutive frames
        # y elements in sparse representation
        # z elements per object
        self.input_size = 4 * 5 * 4

        self.lin1 = nn.Linear(self.input_size, 128)

        self.Alin1 = nn.Linear(128, 128) 
        self.Alin2 = nn.Linear(128, outputs)

        self.Vlin1 = nn.Linear(128, 128)
        self.Vlin2 = nn.Linear(128, 1)

    # Called with one element to determine next action
    def forward(self, x):
        x = x.to(device)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)

        Ax = self.Alin1(x)
        Ax = F.relu(Ax)
        Ax = self.Alin2(Ax)

        Vx = self.Vlin1(x)
        Vx = F.relu(Vx)
        Vx = self.Vlin2(Vx)

        q = Vx + (Ax - Ax.mean())
        return q