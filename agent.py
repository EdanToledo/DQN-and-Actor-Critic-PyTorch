import torch
from torch import nn
import torch.optim
import numpy as np
import random
import torch.nn.functional as F
import os
from utils import Replay_Memory, Transition, Prioritized_Replay_Memory, TransitionPolicy
from torch.distributions import Categorical, Normal
from models import ActorCriticContinuous, ActorCritic, DQN, ICM


class QAgent:

    def __init__(self, epsilon_start, epsilon_end, epsilon_anneal, nb_actions, learning_rate, gamma, batch_size,
                 replay_memory_size, hidden_size, model_input_size, use_PER,use_ICM):

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
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.learning_rate)

        self.use_PER = use_PER
        if use_PER:
            self.replay = Prioritized_Replay_Memory(replay_memory_size)
        else:
            self.replay = Replay_Memory(replay_memory_size)

        self.loss_function = torch.nn.MSELoss()
        self.use_ICM = use_ICM
        if use_ICM:
            self.icm = ICM(model_input_size,nb_actions)

    # Get the current epsilon value according to the start/end and annealing values
    def get_epsilon(self):
        eps = self.epsilon_end
        if self.step_no < self.epsilon_anneal_over_steps:
            eps = self.epsilon_start - self.step_no * \
                ((self.epsilon_start - self.epsilon_end) /
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
        if self.use_ICM:
            self.icm.optimizer.zero_grad()
            forward_loss = self.icm.get_forward_loss(state,action,next_state)
            inverse_loss = self.icm.get_inverse_loss(state,action,next_state)
            icm_loss = (1-self.icm.beta)*inverse_loss.mean()+self.ICM.beta*forward_loss.mean()


        td_estimates = self.policy(state).gather(1, action).squeeze()

        td_targets = reward + (1 - done.float()) * self.gamma * \
            self.target(next_state).max(1)[0].detach_()

        if self.use_PER:

            loss = (torch.tensor(ImportanceSamplingWeights, device=self.device) * self.loss_function(
                td_estimates, td_targets)).sum() * self.loss_function(td_estimates, td_targets)

            errors = td_estimates - td_targets
            self.replay.batch_update(batch_index, errors.data.numpy())
        else:
            loss = self.loss_function(td_estimates, td_targets)

        if self.use_ICM:
            loss = self.icm.lambda_weight*loss + icm_loss

        loss.backward()

        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

        if self.use_ICM:
            self.icm.optimizer.step()

        self.optimizer.step()
        
        return loss.item()

    # set target net parameters to policy net parameters
    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    # save model
    def save(self, path, name):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, os.path.join(path, name + ".pt"))
        torch.save(self.policy.state_dict(), filename)

    # load a model
    def load(self, path):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, path)
        self.policy.load_state_dict(torch.load(filename))

    # store experience in replay memory
    def cache(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

class ActorCriticAgent:

    def __init__(self, nb_actions, learning_rate, gamma, hidden_size, model_input_size, entropy_coeff_start, entropy_coeff_end, entropy_coeff_anneal, continuous):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = nb_actions

        self.gamma = gamma

        self.continuous = continuous

        self.learning_rate = learning_rate

        self.entropy_coefficient_start = entropy_coeff_start
        self.entropy_coefficient_end = entropy_coeff_end
        self.entropy_coefficient_anneal = entropy_coeff_anneal

        self.step_no = 0
        if self.continuous:
            self.model = ActorCriticContinuous(hidden_size=hidden_size,
                                               inputs=model_input_size, outputs=nb_actions).to(self.device)
        else:
            self.model = ActorCritic(hidden_size=hidden_size,
                                     inputs=model_input_size, outputs=nb_actions).to(self.device)

        self.hidden_size = hidden_size
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)

        self.loss_function = torch.nn.MSELoss()

        self.memory = []

    # Get the current entropy coefficient value according to the start/end and annealing values
    def get_entropy_coefficient(self):
        entropy = self.entropy_coefficient_end
        if self.step_no < self.entropy_coefficient_anneal:
            entropy = self.entropy_coefficient_start - self.step_no * \
                ((self.entropy_coefficient_start - self.entropy_coefficient_end) /
                 self.entropy_coefficient_anneal)
        return entropy

    # select an action with policy
    def select_action(self, state):
        self.step_no += 1

        if self.continuous:
            action_mean, action_dev, state_value = self.model(state)
            action_dist = Normal(action_mean, action_dev)
        else:
            action_probs, state_value = self.model(state)
            action_dist = Categorical(action_probs)

        return action_dist, state_value

    def update_model(self):

        Gt = torch.tensor(0)

        policy_losses = []
        value_losses = []
        entropy_loss = []
        returns = []

        # calculate the true value using rewards returned from the environment
        for (_, reward, _, _) in self.memory[::-1]:
            # calculate the discounted value
            Gt = reward + self.gamma * Gt

            returns.insert(0, Gt)

        returns = torch.tensor(returns)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for (action_prob, _, state_value, entropy), Gt in zip(self.memory, returns):

            advantage = Gt.item() - state_value.item()

            # calculate actor (policy) loss
            policy_losses.append((-action_prob * advantage).mean())

            # calculate critic (value) loss using model loss function
            value_losses.append(self.loss_function(
                state_value, Gt.unsqueeze(0)))

            entropy_loss.append(-entropy)

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).mean() + \
            torch.stack(value_losses).mean() + self.get_entropy_coefficient() * \
            torch.stack(entropy_loss).mean()

        loss.backward()

        self.optimizer.step()

        self.memory = []

        return loss.item()

    # save model
    def save(self, path, name):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, os.path.join(path, name + ".pt"))
        torch.save(self.model.state_dict(), filename)

    # load a model
    def load(self, path):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, path)
        self.model.load_state_dict(torch.load(filename))

    def cache(self, action_prob, reward, state_value, entropy):
        self.memory.append(TransitionPolicy(
            action_prob, reward, state_value, entropy))

class ActorCriticAgentUsingICM:

    def __init__(self, nb_actions, learning_rate, gamma, hidden_size, model_input_size, entropy_coeff_start, entropy_coeff_end, entropy_coeff_anneal, continuous):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = nb_actions

        self.gamma = gamma

        self.continuous = continuous

        self.learning_rate = learning_rate

        self.entropy_coefficient_start = entropy_coeff_start
        self.entropy_coefficient_end = entropy_coeff_end
        self.entropy_coefficient_anneal = entropy_coeff_anneal

        self.step_no = 0
        if self.continuous:
            self.model = ActorCriticContinuous(hidden_size=hidden_size,
                                               inputs=model_input_size, outputs=nb_actions).to(self.device)
        else:
            self.model = ActorCritic(hidden_size=hidden_size,
                                     inputs=model_input_size, outputs=nb_actions).to(self.device)

        self.hidden_size = hidden_size
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        self.loss_function = torch.nn.MSELoss()

        self.memory = []

        self.ICM = ICM(model_input_size,nb_actions)
        self.ICM.train()
        

    # Get the current entropy coefficient value according to the start/end and annealing values
    def get_entropy_coefficient(self):
        entropy = self.entropy_coefficient_end
        if self.step_no < self.entropy_coefficient_anneal:
            entropy = self.entropy_coefficient_start - self.step_no * \
                ((self.entropy_coefficient_start - self.entropy_coefficient_end) /
                 self.entropy_coefficient_anneal)
        return entropy

    # select an action with policy
    def select_action(self, state):
        self.step_no += 1

        if self.continuous:
            action_mean, action_dev, state_value = self.model(state)
            action_dist = Normal(action_mean, action_dev)
        else:
            action_probs, state_value = self.model(state)
            action_dist = Categorical(action_probs)

        return action_dist, state_value

    def update_model(self):

        Gt = torch.tensor(0)

        policy_losses = []
        forward_losses = []
        inverse_losses = []
        value_losses = []
        entropy_loss = []
        returns = []

        # calculate the true value using rewards returned from the environment
        for (_, reward, _, _,_,_,_) in self.memory[::-1]:
            # calculate the discounted value
            Gt = reward + self.gamma * Gt

            returns.insert(0, Gt)

        returns = torch.tensor(returns)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for (action_prob, _, state_value, entropy,state,next_state,action), Gt in zip(self.memory, returns):

            advantage = Gt.item() - state_value.item()

            # calculate actor (policy) loss
            policy_losses.append((-action_prob * advantage).mean())

            # calculate critic (value) loss using model loss function
            value_losses.append(self.loss_function(
                state_value, Gt.unsqueeze(0)))

            entropy_loss.append(-entropy)

            forward_losses.append(self.ICM.get_forward_loss(state,action,next_state))
            inverse_losses.append(self.ICM.get_inverse_loss(state,action,next_state))



        # reset gradients
        self.optimizer.zero_grad()
        self.ICM.optimizer.zero_grad()
        # sum up all the values of policy_losses and value_losses
        icm_loss = (1-self.ICM.beta)*torch.stack(inverse_losses).mean()+self.ICM.beta*torch.stack(forward_losses).mean()

        loss = self.ICM.lambda_weight*(torch.stack(policy_losses).mean() + \
            torch.stack(value_losses).mean() + self.get_entropy_coefficient() * \
            torch.stack(entropy_loss).mean()) + icm_loss

        loss.backward()

        self.optimizer.step()
        self.ICM.optimizer.step()
        self.memory = []

        return loss.item()

    # save model
    def save(self, path, name):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, os.path.join(path, name + ".pt"))
        torch.save(self.model.state_dict(), filename)

    # load a model
    def load(self, path):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, path)
        self.model.load_state_dict(torch.load(filename))

    def cache(self, action_prob, reward, state_value, entropy,state,next_state,action):
        self.memory.append((
            action_prob, reward, state_value, entropy,state,next_state,action))
