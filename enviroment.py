import gym
from agent import agent
import numpy as np
from torchvision import transforms as T
import torch
import matplotlib.pyplot as plt
import wandb

# initiate wandb logging
wandb.init(project='DQN-Cartpole', entity='edan')
#choose device for pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Some parameters
MAX_NUMBER_OF_STEPS = 500
EPISODES_TO_TRAIN = 500
START_RENDERING = 500


def play(agent,env, number_of_steps=MAX_NUMBER_OF_STEPS, number_of_episodes=EPISODES_TO_TRAIN):
    
    # loop chosen number of episodes
    for ep_number in range(number_of_episodes):
        # Get initial state observation and format it into a tensor
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32,device = device)
        
        # For reward stat checking
        reward_tot = 0
        # loop max number of steps chosen - breaks if episode finishes before
        for step_no in range(number_of_steps):
            # start rendering if episode number is above chosen - makes training faster by not rendering
            if ep_number > START_RENDERING:
                env.render()

            # log epsilon value of agent
            wandb.log({"epsilon": agent.get_epsilon()})

            # select action
            action = agent.select_action(state)
            # make a step with action in enviroment
            next_state, reward, done, info = env.step(action.item())

            reward_tot += reward

            # if done minus a reward
            if done:
                reward = -1

            # format reward and next_state into tensors
            reward = torch.tensor(reward, dtype=torch.float32,device = device)
            next_state = torch.tensor(next_state, dtype=torch.float32,device = device)

            # store step in agent's replay memory
            agent.cache(state, action, reward, next_state, torch.tensor(done,device = device))

            # update model every step
            loss = agent.update_model()

            # log loss
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

    agent.save("models", "cartpole")
    env.close()


if __name__ == "__main__":
    env = gym.make("CartPole-v1").unwrapped
    
    a = agent(epsilon_start=0.95, epsilon_end=0.001, epsilon_anneal=10000, nb_actions=env.action_space.n, learning_rate=0.0005,
              gamma=0.95, batch_size=3000, replay_memory_size=100000, hidden_size=256, inputs=4)

    
    play(a,env=env)
