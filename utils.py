from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done"))

# Simple Replay Memory
class Replay_Memory:

    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            return None
        return random.sample(self.memory, batch_size)

# SumTree data structure, each node has a value equal to the sum of their children
# Used to store priority value of data points as well as the data in leaf nodes
class SumTree(object):

    

    def __init__(self, size):
        self.size = size  # for all priority values
        self.tree = np.zeros(2 * size - 1) # size of tree - for priority values
        self.data = np.zeros(size, dtype=object)  # size of leaf nodes - for actual data
        self.data_pointer = 0 # current pointer in data - if pointer exceeds size of data- it wraps around to zero
        

    def add(self, priority, data):

        # Index in tree representing position of data that data pointer is pointing to - this is for priority score
        tree_index = self.data_pointer + self.size - 1

        self.data[self.data_pointer] = data  # Add data in current position to list

        self.update(tree_index, priority)  # Update priority in tree with the given priority

        self.data_pointer += 1 # incremenet the data pointer since we have added data to th current position

        if self.data_pointer >= self.size:  # Reset data pointer when it exceeds memory capacity so that it rewrites old data
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # Change in priority of new data's priority 
        change = priority - self.tree[tree_index]

        # set priority to given priority
        self.tree[tree_index] = priority

        # then propagate the change through tree so that the general sum tree property remains correct
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, priority_value):

        parent_index = 0
        while True:     # the while loop is faster than the method in the reference code
            child_left_index = 2 * parent_index + 1         # this node's left and right kids
            child_right_index = child_left_index + 1
            if child_left_index >= len(self.tree):        # reach bottom, end search
                leaf_index = parent_index
                break
            else:       # downward search, always search for a higher priority node
                if priority_value <= self.tree[child_left_index]:
                    parent_index = child_left_index
                else:
                    priority_value -= self.tree[child_left_index]
                    parent_index = child_right_index

        data_index = leaf_index - self.size + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # the root - the sum of all priority values


class Prioritized_Replay_Memory(object):

    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, size):
        self.tree = SumTree(size)

    def push(self, *args):
        max_priority = np.max(self.tree.tree[-self.tree.size:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        self.tree.add(max_priority, Transition(*args))   # set the max priority for new priority

    def sample(self, batch_size):
        batch_index, batch_memory, ImportanceSamplingWeights = np.empty((batch_size), dtype=np.int32), np.empty((batch_size),dtype=object), np.empty((batch_size))
        priority_segment = self.tree.total_priority / batch_size 
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        for i in range(batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            random_priority_value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(random_priority_value)
            prob = priority / self.tree.total_priority
           
            ImportanceSamplingWeights[i] = np.power(prob*self.tree.size, -self.beta)
            batch_index[i], batch_memory[i] = index, data
        
        ImportanceSamplingWeights /= ImportanceSamplingWeights.max()

        return batch_index, batch_memory, ImportanceSamplingWeights

    def batch_update(self, tree_indexes, errors):
        abs_errors = np.abs(errors)
        abs_errors += self.epsilon  
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for tree_index, priority in zip(tree_indexes, ps):
            self.tree.update(tree_index, priority)
