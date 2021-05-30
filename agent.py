import torch
import numpy as np

from buffer import BaseBuffer, UniformBuffer
from network import DRQN
from copy import deepcopy
from torch.autograd import Variable

class DRQNAgent(object):
    def __init__(self, capacity, n_action, state_shape, state_dtype, hidden_shape, gamma, batch_size, device):
        self.capacity = capacity
        self.n_action = n_action
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.hidden_shape = hidden_shape
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.drqn_net = DRQN(n_action = self.n_action, hidden_dim=self.hidden_shape[2]).to(self.device)
        #self.targetnet = deepcopy(self.drqn_net).to(self.device)
        self.buffer = UniformBuffer(capacity = self.capacity,
                                    state_shape = self.state_shape,
                                    state_dtype = self.state_dtype,
                                    )

    def e_greedy_policy(self, state, hidden_state, epsilon):
        if np.random.random() < epsilon:
            _, new_hidden = self.drqn_net(state, hidden_state)
            return np.random.randint(0, self.n_action), new_hidden
        else:
            return self.greedy_policy(state, hidden_state)

    def greedy_policy(self, state, hidden_state):
        values, new_hidden = self.drqn_net(state, hidden_state)
        action = torch.argmax(values, dim = 1)

        return action, new_hidden

    def loss(self,batch_size):
        batch = self.buffer.sample(batch_size)
        batch = self.batch_to_torch(batch, self.device)

        #with torch.no_grad():
        #    next_values = self.targetnet(batch.state).to(self.device)
        #    next_values = torch.max(next_values, dim = 1, keepdim = True)[0]

        lstm_hidden_h = Variable(torch.zeros(1, 1, 16).float()).to(self.device)
        lstm_hidden_c = Variable(torch.zeros(1, 1, 16).float()).to(self.device)

        current_values, _ = self.drqn_net(batch.state, (lstm_hidden_h, lstm_hidden_c))
        current_values = current_values.gather(1, batch.action)

        next_values = torch.max(current_values, dim = 1, keepdim = True)[0].detach().clone()
        
        target_value = batch.reward + next_values * (1 - batch.terminal) * self.gamma

        td_error = torch.nn.functional.smooth_l1_loss(current_values, target_value).to(self.device)

        return td_error

    @staticmethod
    def batch_to_torch(batch, device):
        return BaseBuffer.Transition(*(torch.from_numpy(x).type(dtype).to(device) for x, dtype in zip(batch,(torch.float, torch.long, torch.float32, torch.float32, torch.float32))))
        
    @staticmethod
    def state_to_torch(state, device):
        return torch.from_numpy(state).float().to(device)


