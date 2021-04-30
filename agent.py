import torch
from torch import nn
import torch.optim
import numpy as np
import random
import torch.nn.functional as F
from collections import namedtuple, deque
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, inputs, hidden_size, outputs):
        super(DQN, self).__init__()
        # Simple feed forward NN
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, outputs))

    def forward(self, x):
        return self.net(x)


class agent:

    def __init__(self, epsilon_start, epsilon_end, epsilon_anneal, nb_actions, learning_rate, gamma, batch_size, replay_memory_size, hidden_size, inputs):

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_anneal_over_steps = epsilon_anneal

        self.num_actions = nb_actions

        self.gamma = gamma

        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.step_no = 0

        self.policy = DQN(hidden_size=hidden_size,
                          inputs=inputs, outputs=nb_actions).to(device)
        self.target = DQN(hidden_size=hidden_size,
                          inputs=inputs, outputs=nb_actions).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.hidden_size = hidden_size
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate)

        self.replay = replay_memory(replay_memory_size)

        self.loss_function = torch.nn.SmoothL1Loss()

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
            return torch.tensor([random.randrange(self.num_actions)], device=device, dtype=torch.long)

    # update the model according to one step td targets
    def update_model(self):
        batch = self.replay.sample(self.batch_size)

        if batch == None:
            return

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

    # store experience in replay memory
    def cache(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done"))


# simple replay memory class

class replay_memory:

    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            return None
        return random.sample(self.memory, batch_size)
