
import torch
from torch import nn
import torch.optim
import numpy as np
import random
import torch.nn.functional as F
import os
from utils import Replay_Memory, Transition, Prioritized_Replay_Memory, TransitionPolicy
from torch.distributions import Categorical, Normal


class DQN(nn.Module):

    def __init__(self, inputs, hidden_size, outputs):
        super(DQN, self).__init__()
        # Simple feed forward NN
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, outputs))

    def forward(self, x):
        return self.net(x)


class ActorCriticContinuous(nn.Module):
    def __init__(self, inputs, hidden_size, outputs):
        super(ActorCriticContinuous, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, outputs)
        )

        # self.log_deviations = nn.Parameter(torch.full((outputs,), 0.2))
        self.dev = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, outputs)
        )

        self.critic = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        means = self.policy(x)
        state_value = self.critic(x)
        dev = torch.clamp(self.dev(x).exp(), 1e-3, 50)
        # dev = torch.clamp(self.log_deviations.exp(), 1e-3, 50)
        return means, dev, state_value


class ActorCritic(nn.Module):
    def __init__(self, inputs, hidden_size, outputs):
        super(ActorCritic, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, outputs),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        policy = self.policy(x)
        state_value = self.critic(x)

        return policy, state_value
