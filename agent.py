import torch
from torch import nn
import torch.optim
import numpy as np
import random
import torch.nn.functional as F
import os
from utils import Replay_Memory, Transition, Prioritized_Replay_Memory


class DQN(nn.Module):

    def __init__(self, inputs, hidden_size, outputs):
        super(DQN, self).__init__()
        # Simple feed forward NN
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, outputs))

    def forward(self, x):
        return self.net(x)


class agent:

    def __init__(self, epsilon_start, epsilon_end, epsilon_anneal, nb_actions, learning_rate, gamma, batch_size, replay_memory_size, hidden_size, model_input_size, use_PER):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_anneal_over_steps = epsilon_anneal

        self.num_actions = nb_actions

        self.gamma = gamma

        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.step_no = 0

        self.policy = DQN(hidden_size=hidden_size,
                          inputs=model_input_size, outputs=nb_actions).to(self.device)
        self.target = DQN(hidden_size=hidden_size,
                          inputs=model_input_size, outputs=nb_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.hidden_size = hidden_size
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate,weight_decay=0.01)

        self.use_PER = use_PER
        if use_PER:
            self.replay = Prioritized_Replay_Memory(replay_memory_size)
        else:
            self.replay = Replay_Memory(replay_memory_size)

        self.loss_function = torch.nn.MSELoss()

    # Get the current epsilon value according to the start/end and annealing values
    def get_epsilon(self):
        eps = self.epsilon_end
        if self.step_no < self.epsilon_anneal_over_steps:
            eps = self.epsilon_start - self.step_no * \
                ((self.epsilon_start-self.epsilon_end) /
                 self.epsilon_anneal_over_steps)
        return eps

    # select an action with epsilon greedy
    def select_action(self, state):
        self.step_no += 1
        if np.random.uniform() > self.get_epsilon():
            with torch.no_grad():
                return torch.argmax(self.policy(state)).view(1)
        else:
            return torch.tensor([random.randrange(self.num_actions)], device=self.device, dtype=torch.long)

    # update the model according to one step td targets
    def update_model(self):
        if self.use_PER:
            batch_index, batch, ImportanceSamplingWeights = self.replay.sample(
                self.batch_size)
        else:
            batch = self.replay.sample(self.batch_size)

        batch_tuple = Transition(*zip(*batch))

        state = torch.stack(batch_tuple.state)
        action = torch.stack(batch_tuple.action)
        reward = torch.stack(batch_tuple.reward)
        next_state = torch.stack(batch_tuple.next_state)
        done = torch.stack(batch_tuple.done)

        self.optimizer.zero_grad()

        td_estimates = self.policy(state).gather(1, action).squeeze()

        td_targets = reward+(1-done.float())*self.gamma * \
            self.target(next_state).max(1)[0].detach_()

        if self.use_PER:
            
            loss = (torch.tensor(ImportanceSamplingWeights) * \
                self.loss_function(td_estimates, td_targets)).mean()

            errors = td_estimates - td_targets
            self.replay.batch_update(batch_index, errors.data.numpy())
        else:
            loss = self.loss_function(td_estimates, td_targets)

        loss.backward()

        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss.item()

    # set target net parameters to policy net parameters
    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    # save model
    def save(self, path, name):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, os.path.join(path, name+".pt"))

        torch.save({
            'model_state_dict': self.policy.state_dict()
        }, filename)

    # load a model
    def load(self, path):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, path)
        self.policy.load_state_dict(torch.load(filename)['model_state_dict'])
        self.policy.eval()

    # store experience in replay memory

    def cache(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)
