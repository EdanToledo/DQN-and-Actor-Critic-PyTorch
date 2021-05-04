import gym
from agent import agent
import numpy as np
from torchvision import transforms as T
import torch
import matplotlib.pyplot as plt
import wandb

# initiate wandb logging
wandb.init(project='DQN-Cartpole', entity='edan')

# Some parameters
MAX_NUMBER_OF_STEPS = 500
EPISODES_TO_TRAIN = 700
START_RENDERING = 400
LOG_VIDEO = False

# returns a single rgb array of a frame of environment
def get_frame(env):
    return env.render(mode="rgb_array").transpose(2, 0, 1)

# trains agent
def train(agent, env, number_of_steps=MAX_NUMBER_OF_STEPS, number_of_episodes=EPISODES_TO_TRAIN):

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

            if done:
                reward-=1

            reward_tot += reward
            
            # format reward and next_state into tensors
            reward = torch.tensor(reward, dtype=torch.float32, device=agent.device)
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=agent.device)

            # store step in agent's replay memory
            agent.cache(state, action, reward, next_state,
                        torch.tensor(done, device=agent.device))

            # update model every step
            if agent.step_no > agent.batch_size:
                loss = agent.update_model()            
                wandb.log({"loss": loss})

            # set current state as next_state
            state = next_state

            # if done finish episode
            if done:
                break

        # every few episodes - update target network with current policy weights
        if ep_number % 10 == 0:
            agent.update_target()

        # logging
        wandb.log({"reward": reward_tot})
        wandb.log({"episode": ep_number})
        if LOG_VIDEO:
            wandb.log({"Episode Simulation Render": wandb.Video(np.stack(frames), fps=50, format="gif")})

    agent.save("models", "cartpole")
    env.close()

# Plays and renders a number of episodes without training
def eval(agent, env, number_of_episodes):
    # loop chosen number of episodes
    for ep_number in range(number_of_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device)

        done = False
        while not done:
            env.render()
            # select action
            action = agent.select_action(state)
            next_state, _, done, _ = env.step(action.item())
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=agent.device)

            state = next_state

    env.close()


if __name__ == "__main__":
    env = gym.make("CartPole-v1").unwrapped

    a = agent(epsilon_start=0.95, epsilon_end=0.01, epsilon_anneal=12000, nb_actions=env.action_space.n, learning_rate=0.001,
              gamma=1, batch_size=64, replay_memory_size=100000, hidden_size=128, model_input_size=4,use_PER=True)

    train(a, env=env)

    #a.load("models/cartpole.pt")

    #eval(a,env=env,number_of_episodes=20)
