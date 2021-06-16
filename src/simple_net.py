import torch
import torch.nn as nn
import torch.nn.functional as F
from rational.torch import Rational

# n_features = [60]
class Network(nn.Module):

    def __init__(self, input_size=32, output_size=12, **kwargs):
        super().__init__()


        # self._h1 = nn.Linear(input_size, n_features[0])
        # self._h3 = nn.Linear(n_features[0], output_size)
        self._h3 = nn.Linear(input_size, output_size)

        # nn.init.xavier_uniform_(self._h1.weight,
        #                         gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))


        # self.act_func1 = Rational()
        # self.act_func2 = Rational()
        # self.act_func1 = F.leaky_relu
        # self.act_func2 = F.leaky_relu

    def forward(self, state):
        # x1 = self._h1(state)
        # h = self.act_func1(x1)
        # x2 = self._h2(h)
        # h = self.act_func2(x2)
        return F.log_softmax(self._h3(state))
        # return F.log_softmax(self._h3(h))
