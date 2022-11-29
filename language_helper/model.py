
"""
Absorbing Markov Model takes current state as input,
and output the next probable state

For example:
    - input: hi, how, are, you, doing
    
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import FloatTensor, randn, zeros

class MarkovNgrams:
    def __init__(state):
        
        pass

class RNN(nn.Module):
    """
    simple RNN network with 1 layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden
    
    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


