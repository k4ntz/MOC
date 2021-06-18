import gym
import torch
import random
import math

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple

from dqn.dqn_networks import LinearNN
from dqn.dqn_networks import SPACEDuelCNN
from dqn.dqn_networks import DuelCNN

### replay memory stuff
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class Agent:
    def __init__(self, batch_size, gamma, eps_start, eps_end, eps_decay, 
                    lr, n_actions, memory_min_size, device, log_steps,  use_space, cfg):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr
        self.n_actions = n_actions
        self.memory_min_size = memory_min_size

        self.device = device
        self.log_steps = log_steps
        
        self.use_space = use_space

        # init neural nets
        self.policy_net = None
        self.target_net = None

        n_inputs = 3 if cfg.train.use_enemy else 2
        n_features = 4 if not cfg.train.use_zwhat else 36

        if self.use_space:
            if cfg.train.cnn_features:
                self.policy_net = SPACEDuelCNN(n_actions).to(device)
                self.target_net = SPACEDuelCNN(n_actions).to(device)
            else:
                self.policy_net = LinearNN(n_inputs, n_features, n_actions).to(device)
                self.target_net = LinearNN(n_inputs, n_features, n_actions).to(device)
        else:
            self.policy_net = DuelCNN(64, 64, n_actions).to(device)
            self.target_net = DuelCNN(64, 64, n_actions).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)


    # function to select action by given state
    def select_action(self, state, global_step, logger):
        sample = random.random()
        eps_threshold = self.eps_start
        if global_step > self.memory_min_size:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * (global_step - self.memory_min_size) / self.eps_decay)
        # log eps_treshold
        if global_step % self.log_steps == 0:
            logger.log_eps(eps_threshold, global_step)
        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)


    # function to train model  
    def optimize_model(self, memory, total_max_q, total_loss, logger, global_step):
        if len(memory) < self.batch_size:
            return total_max_q, total_loss
        transitions = memory.sample(self.batch_size)
        """
        zip(*transitions) unzips the transitions into
        Transition(*) creates new named tuple
        batch.state - tuple of all the states (each state is a tensor)
        batch.next_state - tuple of all the next states (each state is a tensor)
        batch.reward - tuple of all the rewards (each reward is a float)
        batch.action - tuple of all the actions (each action is an int)    
        """
        batch = Transition(*zip(*transitions))
        
        # Convert them to tensors
        state = torch.from_numpy(np.array(batch.state)).float().to(self.device)
        next_state = torch.from_numpy(np.array(batch.next_state)).float().to(self.device)
        action = torch.cat(batch.action, dim=0).to(self.device).squeeze(1)
        reward = torch.cat(batch.reward, dim=0).to(self.device)
        done = torch.tensor(batch.done, dtype=torch.float, device=self.device)

        # Make predictions
        state_q_values = self.policy_net(state)
        next_states_q_values = self.policy_net(next_state)
        next_states_target_q_values = self.target_net(next_state)
        # Find selected action's q_value
        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # Get indice of the max value of next_states_q_values
        # Use that indice to get a q_value from next_states_target_q_values
        # We use greedy for policy So it called off-policy
        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        # Use Bellman function to find expected q value
        expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

        # Calc loss with expected_q_value and q_value
        loss = F.mse_loss(selected_q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log loss and max q
        max_q = torch.max(state_q_values).item()
        total_max_q += max_q
        with torch.no_grad():
            total_loss += loss
        # log metrics for last log_steps and reset
        if global_step % self.log_steps == 0:
            logger.log_max_q(total_max_q/self.log_steps, global_step)
            total_max_q = 0
            logger.log_loss(total_loss/self.log_steps, global_step)
            total_loss = 0
        return total_max_q, total_loss



