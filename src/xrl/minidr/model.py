import torch
import torch.nn as nn
import torch.nn.functional as F


# with preprocessed meaningful features
class WorldPredictor(nn.Module):
    def __init__(self, input): 
        super(WorldPredictor, self).__init__()
        # transition layers
        # Take in previous state s_{t-1} and action a_{t-1} and predicts the next state s_t , represents p(s_t|s_{t-1}, a_{t-1})
        self.transition = nn.Sequential(
            nn.Linear(input + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input),
            nn.ReLU(),
        )

        # reward prediction layer
        # Take in state s_{t} and predicts reward r_{t}
        self.reward = nn.Sequential(
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

    # gets last state together concat with action
    # predicts next state 
    # then predicts its reward
    def forward(self, last_state, action):
        state = self.transition(torch.cat((last_state, action.unsqueeze(2)), dim=2))
        reward = self.reward(state)
        return state, reward


# policy model
class Policy(nn.Module):
    def __init__(self, input, hidden, actions): 
        super(Policy, self).__init__()
        print("Policy net has", input, "input nodes,", hidden, "hidden nodes and", actions, "output nodes")
        self.h = nn.Linear(input, hidden)
        self.out = nn.Linear(hidden, actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.h(x))
        return F.softmax(self.out(x), dim=1)