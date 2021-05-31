import torch
import torch.nn as nn
import numpy as np

class DRQN(nn.Module):
    def __init__(self, n_action, hidden_dim, lstm_input_dim=144):
        super(DRQN, self).__init__()
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = hidden_dim
        self.n_layer = 1
        self.output_dim = n_action

        self.conv_net = nn.Sequential(
                            nn.Conv2d(3, 8, (7,3), stride=3),
                            nn.ReLU(),
                            nn.Conv2d(8, 8, (5,3), stride=3),
                            nn.ReLU(),
                            nn.Conv2d(8, 16, (5,3), stride=2),
                            nn.ReLU(),
                            nn.Conv2d(16, 16, (5,3), stride=2),
                            nn.ReLU()
                                    )

        self.lstm_net = nn.LSTM(
                            input_size=self.lstm_input_dim,
                            hidden_size=self.lstm_hidden_dim,
                            num_layers=self.n_layer                          
                                )

        self.fc = nn.Sequential(
                        nn.Linear(self.lstm_hidden_dim, self.output_dim),
                                )

    def forward(self, state, hidden_state):
        state = state.view(-1,3,210,160) / 255.0
        #state = state.permute(0,3,1,2)
        conv_out = self.conv_net(state)
        flatten = conv_out.view(conv_out.size(0), -1).unsqueeze(0)
        #flatten = torch.flatten(conv_out, start_dim=1, end_dim=-1).unsqueeze(0)
        lstm_out, new_hidden = self.lstm_net(flatten, hidden_state)
        lstm_out = lstm_out.view(-1,self.lstm_hidden_dim)
        output = self.fc(lstm_out)

        return output, new_hidden

