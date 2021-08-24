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
            F.relu(),
            nn.Linear(128, 128),
            F.relu(),
            nn.Linear(128, input),
            F.relu()
        )

        # reward prediction layer
        # Take in state s_{t} and predicts reward r_{t}
        self.reward = nn.Sequential(
            nn.Linear(input, 128),
            F.relu(),
            nn.Linear(128, 1),
            F.relu()
        )

    # gets last state together concat with action
    # predicts next state 
    # then predicts its reward
    def forward(self, last_state):
        state = self.transition(last_state)
        reward = self.reward(state)
        return state, reward