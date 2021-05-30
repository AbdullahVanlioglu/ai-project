import torch.nn as nn
import numpy as np

class DRQN(nn.Module):
    def __init__(self, n_action, lstm_input_dim=256, hidden_dim=16):
        super(DRQN, self).__init__()
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = hidden_dim
        self.n_layer = 1
        self.output_dim = n_action


        self.conv_net = nn.Sequential(
                            nn.Conv2d(3, 8, (7,3), stride=2),
                            nn.ReLU(),
                            nn.Conv2d(8, 16, (5,3), stride=2),
                            nn.ReLU(),
                            nn.Conv2d(16, 16, (5,3), stride=2),
                            nn.ReLU(),
                            nn.Conv2d(16, 32, (5,3), stride=2),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, (3,3), stride=2),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, (3,3), stride=1),
                            nn.ReLU()
                                    )

        self.lstm_net = nn.LSTM(
                            input_size=self.lstm_input_dim,
                            hidden_size=self.lstm_hidden_dim,
                            num_layers=self.n_layer                          
                                )

        self.fc = nn.Sequential(
                        nn.Linear(self.lstm_hidden_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, self.output_dim),
                                )

    
    def forward(self, state, hidden_state):
        state = state.view(-1,3,210,160)
        conv_out = self.conv_net(state)
        flatten = conv_out.view(conv_out.size(0), -1).unsqueeze(1)
        lstm_out, new_hidden = self.lstm_net(flatten, hidden_state)
        hidden_h, hidden_c = new_hidden
        output = self.fc(lstm_out)

        return output, new_hidden

