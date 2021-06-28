import gym
from agent import ActorCriticAgent, QAgent
import numpy as np
from torchvision import transforms as T
import torch
import matplotlib.pyplot as plt
import wandb
from torch.distributions import Categorical
import argparse

# initiate wandb logging
wandb.init(project='DQN-Cartpole', entity='edan')

# Some parameters

LOG_VIDEO = False


# returns a single rgb array of a frame of environment


def get_frame(env):
    return env.render(mode="rgb_array").transpose(2, 0, 1)


# trains agent
def trainQ(agent, env, number_of_steps, number_of_episodes, START_RENDERING, update_frequency):
    # loop chosen number of episodes
    for ep_number in range(number_of_episodes):
        # Get initial state observation and format it into a tensor
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device)

        # For reward stat checking
        reward_tot = 0
        frames = []
        # loop max number of steps chosen - breaks if episode finishes before
        for step_no in range(number_of_steps):
            # start rendering if episode number is above chosen - makes training faster by not rendering
            if ep_number > START_RENDERING:
                env.render()

            if LOG_VIDEO:
                frames.append(get_frame(env))

            # log epsilon value of agent
            wandb.log({"epsilon": agent.get_epsilon()})

            # select action
            action = agent.select_action(state)
            # make a step with action in enviroment
            next_state, reward, done, info = env.step(action.item())

            reward_tot += reward

            # format reward and next_state into tensors
            reward = torch.tensor(
                reward, dtype=torch.float32, device=agent.device)
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=agent.device)

            # store step in agent's replay memory
            agent.cache(state, action, reward, next_state,
                        torch.tensor(done, device=agent.device))

            # update model every step
            if agent.step_no > agent.batch_size:
                loss = agent.update_model()
                wandb.log({"loss": loss})

            if agent.step_no % update_frequency == 0:
                agent.update_target()

            # set current state as next_state
            state = next_state

            # if done finish episode
            if done:
                break

        # logging
        wandb.log({"reward": reward_tot})
        wandb.log({"episode": ep_number})
        if LOG_VIDEO:
            wandb.log({"Episode Simulation Render": wandb.Video(
                np.stack(frames), fps=50, format="gif")})

    agent.save("models", "car")
    env.close()

# trains agent


def trainActor(agent, env, number_of_steps, number_of_episodes, START_RENDERING):
    # loop chosen number of episodes
    for ep_number in range(number_of_episodes):
        # Get initial state observation and format it into a tensor
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device)

        # For reward stat checking
        reward_tot = 0
        # loop max number of steps chosen - breaks if episode finishes before
        for step_no in range(number_of_steps):
            # start rendering if episode number is above chosen - makes training faster by not rendering
            if ep_number > START_RENDERING:
                env.render()

            # select action
            action_dist, state_value = agent.select_action(state)

            action = action_dist.sample()
            # make a step with action in enviroment
            next_state, reward, done, info = env.step(action.item())

            reward_tot += reward

            # format reward and next_state into tensors
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=agent.device)
            reward = torch.tensor(
                reward, dtype=torch.float32, device=agent.device)

            # store step in agent's replay memory
            agent.cache(action_dist.log_prob(action), reward, state_value)

            # set current state as next_state
            state = next_state

            # if done finish episode
            if done:
                break

        loss = agent.update_model()
        wandb.log({"loss": loss})
        wandb.log({"reward": reward_tot})
        wandb.log({"episode": ep_number})

    agent.save("models", "car")
    env.close()


def eval(agent, env, number_of_episodes):
    # loop chosen number of episodes
    for ep_number in range(number_of_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device)

        done = False
        tot_reward = 0
        while not done:
            env.render()
            # select action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            tot_reward += reward
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=agent.device)

            state = next_state

        print(tot_reward)
    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train a agent on gym environments')

    parser.add_argument('--gym_env', "-g", default="CartPole-v1", type=str,
                        help='The name of the gym environment')

    parser.add_argument('--hidden_size', "-hs", default=128, type=int,
                        help='size of the hidden layer')

    parser.add_argument('--gamma', "-gm", default=0.99, type=float,
                        help='The discount factor used by the agent')

    parser.add_argument('--learning_rate', "-lr", default=0.001, type=float,
                        help='The learning rate used by the optimizer')

    parser.add_argument('--epsilon_start', "-es", default=0.9, type=float,
                        help='The starting value for epsilon in epsilon-greedy')

    parser.add_argument('--epsilon_end', "-ee", default=0.001, type=float,
                        help='The ending value for epsilon in epsilon-greedy')

    parser.add_argument('--epsilon_anneal', "-en", default=12000, type=int,
                        help='The number of steps to which the epsilon anneals down')

    parser.add_argument('--batch_size', "-bs", default=1500, type=int,
                        help='Batch size to use in DQN')

    parser.add_argument('--replay_memory_size', "-re", default=100000, type=int,
                        help='Size of replay memory')

    parser.add_argument('--use_PER', "-up",
                        action='store_true', help='Use prioritised replay memory in DQN')

    parser.add_argument('--use_DQN', "-ud",
                        action='store_true', help='Use DQN agent instead of advantage-critic')

    parser.add_argument('--MAX_NUMBER_OF_STEPS', "-ms", default=501, type=int,
                        help='The max number of steps per episode')

    parser.add_argument('--EPISODES_TO_TRAIN', "-et", default=1000, type=int,
                        help='The number of episodes to train')

    parser.add_argument('--START_RENDERING', "-sr", default=900, type=int,
                        help='The number of episodes to train before rendering - used for training speed up')

    parser.add_argument('--update_frequency', "-uf", default=1000, type=int,
                        help='The number of steps per updating target DQN')

    args = parser.parse_args()

    env = gym.make(args.gym_env)

    if args.use_DQN:

        a = QAgent(epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_anneal=args.epsilon_anneal, nb_actions=env.action_space.n,
                   learning_rate=args.learning_rate,
                   gamma=args.gamma, batch_size=args.batch_size, replay_memory_size=args.replay_memory_size, hidden_size=args.hidden_size,
                   model_input_size=env.observation_space.shape[0], use_PER=args.use_PER)
        trainQ(a, env, args.MAX_NUMBER_OF_STEPS, args.EPISODES_TO_TRAIN,
               args.START_RENDERING, args.update_frequency)
    else:

        a = ActorCriticAgent(nb_actions=env.action_space.n,
                             learning_rate=args.learning_rate,
                             gamma=args.gamma, hidden_size=args.hidden_size,
                             model_input_size=env.observation_space.shape[0])
        trainActor(a, env, args.MAX_NUMBER_OF_STEPS,
                   args.EPISODES_TO_TRAIN, args.START_RENDERING)