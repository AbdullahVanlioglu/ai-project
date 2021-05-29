import torch
import numpy as np

from buffer import BaseBuffer, UniformBuffer
from network import DRQN
from copy import deepcopy

class DRQNAgent(object):
    def __init__(self, capacity, n_action, state_shape, state_dtype, gamma, batch_size, device):
        self.capacity = capacity
        self.n_action = n_action
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.drqn_net = DRQN(n_action = self.n_action, input_dim=self.state_shape[0]).to(self.device)
        self.targetnet = deepcopy(self.drqn_net).to(self.device)
        self.buffer = UniformBuffer(capacity = self.capacity,
                                    state_shape = self.state_shape,
                                    dtype = self.state_dtype
                                    )

    def e_greedy_policy(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_action)
        else:
            return self.greedy_policy(state, self.device)

    def greedy_policy(self, state):
        values = self.drqn_net(state).to(self.device)
        action = torch.argmax(values, dim = 1)

    def loss(self,batch_size):

        batch = self.buffer.sample(batch_size)
        batch = self.batch_to_torch(batch, self.device)

        with torch.no_grad():
            next_values = self.targetnet(batch.next_state).to(self.device)
            next_values = torch.max(next_values, dim = 1, keepdim = True)[0]
        
        target_value = batch.reward + next_values * (1 - batch.terminal) * self.gamma

        current_values = self.drqn_net(batch.state).to(self.device)
        current_values = current_values.gather(1, batch.action)

        td_error = torch.nn.functional.smooth_l1_loss(current_values, target_value).to(self.device)

        return td_error

    @staticmethod
    def batch_to_torch(batch, device):
        return BaseBuffer.Transition(*(torch.from_numpy(x).type(dtype).to(device) for x, dtype in zip(batch,(torch.float, torch.long, torch.float32, torch.float32, torch.float32))))
        
    @staticmethod
    def state_to_torch(state, device):
        return torch.from_numpy(state).float().to(device)


