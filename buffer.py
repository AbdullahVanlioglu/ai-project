import numpy as np

from collections import namedtuple

class BaseBuffer():

    Transition = namedtuple("Transition", 
                            "state action reward next_state terminal")

    def __init__(self, capacity, state_shape, state_dtype):

        self.capacity = capacity

        self.transition_info = self.Transition(
            {"shape": state_shape, "dtype": state_dtype},
            {"shape": (1,), "dtype": np.int64},
            {"shape": (1,), "dtype": np.float32},
            {"shape": state_shape, "dtype": state_dtype},
            {"shape": (1,), "dtype": np.float32},
        )

        self.buffer = self.Transition(
            *(np.zeros((capacity, *x["shape"]),dtype = x["dtype"])
        for x in self.transition_info)
        )

    def __len__(self):
        return self.capacity

    def push(self, transition, *args, **kwargs):
        raise NotImplementedError

    def sample(self, batchsize, *args, **kwargs):
        raise NotImplementedError


class UniformBuffer(BaseBuffer):

    def __init__(self, capacity, state_shape, state_dtype):
        super().__init__(capacity, state_shape, state_dtype)

        self._cycle = 0
        self.size = 0

    def push(self, transition):
        self.buffer.state[self._cycle] = transition.state
        self.buffer.action[self._cycle] = transition.action
        self.buffer.reward[self._cycle] = transition.reward
        self.buffer.next_state[self._cycle] = transition.next_state
        self.buffer.terminal[self._cycle] = transition.terminal
        self._cycle = (self._cycle+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batchsize, *args):
        if batchsize > self.size:
            return None
        
        idxs = np.random.randint(0, self.size, size = batchsize)
        batch = (self.buffer.state[idxs],
                    self.buffer.action[idxs],
                    self.buffer.reward[idxs],
                    self.buffer.next_state[idxs],
                    self.buffer.terminal[idxs])
        
        return self.Transition(*batch)

