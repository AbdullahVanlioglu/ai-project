import torch.nn as nn
import numpy as np

class DRQN(nn.Module):
    def __init__(self, n_action, input_dim=16, hidden_dim=16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layer = 1
        self.output_dim = n_action


        self.conv_net = nn.Sequential(
                            nn.Conv2d(self.input_dim, 16, 3, stride=1),
                            nn.ReLU(),
                                    )

        self.lstm_net = nn.LSTM(
                            input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layer                          
                                )

        self.fc = nn.Sequential(
                        nn.Linear(self.hidden_dim, 16),
                        nn.ReLU(),
                        nn.Linear(16, self.output_dim),
                                )

    
    def forward(self, x, hidden_state):
        conv_out = self.conv_net(x)
        flat_out = conv_out.view(conv_out.size(0), -1).unsqueeze(1)
        lstm_out, new_hidden = self.lstm_net(flat_out, hidden_state)
        hidden_h, hidden_c = new_hidden
        output = self.fc1(lstm_out)

        return output, new_hidden

