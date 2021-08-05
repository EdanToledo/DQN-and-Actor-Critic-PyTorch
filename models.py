
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

class ICM_Inverse(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(ICM_Inverse,self).__init__()
        self.inverse_net = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size,output_size)
        )
        

    def forward(self,state,next_state):
        input = torch.cat([state,next_state],dim=-1)
        return self.inverse_net(input)

class ICM_Forward(nn.Module):

    def __init__(self,input_size,action_size,hidden_size,output_size):
        super(ICM_Forward,self).__init__()
        self.forward_net = nn.Sequential(
            nn.Linear(input_size+action_size,hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size,output_size)
        )
        self.eye = torch.eye(action_size)

    def forward(self,state,action):
        input = torch.cat([state,self.eye[action].detach()],dim=-1)
        return self.forward_net(input)
        

class ICM_Feature(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size):
        super(ICM_Feature,self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,output_size)
        )

    def forward(self,input): 
        return self.feature_net(input)


class ICM():

    def __init__(self,state_input_size,action_size):
        # self.config = config
        self.read_config()
        self.feature = ICM_Feature(state_input_size,self.hidden_size,self.feature_size)
        self.inverse = ICM_Inverse(self.feature_size*2,self.hidden_size,action_size)
        self.forward = ICM_Forward(self.feature_size,action_size,self.hidden_size,self.feature_size)
        self.model_params = list(self.feature.parameters())+list(self.inverse.parameters())+list(self.forward.parameters())
        self.optimizer = torch.optim.Adam(self.model_params,lr=self.learning_rate)

    def train(self):
        self.feature.train()
        self.inverse.train()
        self.forward.train()
    

    def read_config(self):
        self.scaling_factor = 100
        self.beta = 0.2
        self.lambda_weight = 0.1
        self.learning_rate = 0.001
        self.hidden_size = 16
        self.feature_size = 16

    def get_feature(self,state):
        return self.feature(state)

    def get_predicted_action(self,state, next_state):
        state_feature = self.get_feature(state)
        next_state_feature = self.get_feature(next_state)
        return self.inverse(state_feature,next_state_feature)

    def get_predicted_state(self,state,action):
        state_feature = self.get_feature(state)
        return self.forward(state_feature,action.detach())

    def get_inverse_loss(self,state,action,next_state):
        action = action.unsqueeze(0)
        predicted_action = self.get_predicted_action(state,next_state)
        return F.cross_entropy(predicted_action.unsqueeze(0),action.detach())

    def get_forward_loss(self,state,action,next_state):
        next_state_feature = self.get_feature(next_state)
        predicted_state_feature = self.get_predicted_state(state,action)
        return 0.5*F.mse_loss(predicted_state_feature,next_state_feature)

    def get_intrinsic_reward(self,state,action,next_state):
        with torch.no_grad():
            intrinsic_reward = self.scaling_factor*self.get_forward_loss(state,action,next_state)
            return intrinsic_reward