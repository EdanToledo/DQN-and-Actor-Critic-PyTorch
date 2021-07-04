# Simple implementation of a DQN and Advantage Critic 
Really simple implementation of DQN and Advantage-Critic in pytorch for gym environments. Discrete and Continuous action spaces supported.
# Contents
* agent.py - contains code for agents and models
* utils.py - contains code for replay memory
* environment.py - contains code for training and evaluation agents
# Usage 
To use, simply run the environment.py script using the arguments given below
e.g
```bash
python environment.py -ud
```
# Arguments
```

usage: environment.py [-h] [--gym_env GYM_ENV] [--hidden_size HIDDEN_SIZE] [--gamma GAMMA] [--learning_rate LEARNING_RATE]
                     [--epsilon_start EPSILON_START] [--epsilon_end EPSILON_END] [--epsilon_anneal EPSILON_ANNEAL]
                     [--batch_size BATCH_SIZE] [--replay_memory_size REPLAY_MEMORY_SIZE] [--use_PER] [--use_DQN]
                     [--MAX_NUMBER_OF_STEPS MAX_NUMBER_OF_STEPS] [--EPISODES_TO_TRAIN EPISODES_TO_TRAIN]
                     [--START_RENDERING START_RENDERING] [--update_frequency UPDATE_FREQUENCY]

Train a agent on gym environments

optional arguments:
  -h, --help            show this help message and exit
  --gym_env GYM_ENV, -g GYM_ENV
                        The name of the gym environment
  --hidden_size HIDDEN_SIZE, -hs HIDDEN_SIZE
                        size of the hidden layer
  --gamma GAMMA, -gm GAMMA
                        The discount factor used by the agent
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        The learning rate used by the optimizer
  --epsilon_start EPSILON_START, -es EPSILON_START
                        The starting value for epsilon in epsilon-greedy
  --epsilon_end EPSILON_END, -ee EPSILON_END
                        The ending value for epsilon in epsilon-greedy
  --epsilon_anneal EPSILON_ANNEAL, -en EPSILON_ANNEAL
                        The number of steps to which the epsilon anneals down
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size to use in DQN
  --replay_memory_size REPLAY_MEMORY_SIZE, -re REPLAY_MEMORY_SIZE
                        Size of replay memory
  --use_PER, -up        Use prioritised replay memory in DQN
  --use_DQN, -ud        Use DQN agent instead of advantage-critic
  --MAX_NUMBER_OF_STEPS MAX_NUMBER_OF_STEPS, -ms MAX_NUMBER_OF_STEPS
                        The max number of steps per episode
  --EPISODES_TO_TRAIN EPISODES_TO_TRAIN, -et EPISODES_TO_TRAIN
                        The number of episodes to train
  --START_RENDERING START_RENDERING, -sr START_RENDERING
                        The number of episodes to train before rendering - used for training speed up
  --update_frequency UPDATE_FREQUENCY, -uf UPDATE_FREQUENCY
                        The number of steps per updating target DQN
```
