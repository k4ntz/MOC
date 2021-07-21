import torch
import torch.nn as nn
import torch.nn.functional as F
from rational.torch import Rational


n_features = [48, 48]
class Network(nn.Module):

    def __init__(self, input_size=32, output_size=12, **kwargs):
        super().__init__()

        self._h1 = nn.Linear(input_size, n_features[0])
        self._h2 = nn.Linear(n_features[0],  n_features[1])
        self._h3 = nn.Linear(n_features[1], output_size)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        x = F.leaky_relu(self._h1(x))
        x = F.leaky_relu(self._h2(x))
        return F.log_softmax(self._h3(x))
