
"""
Absorbing Markov Model takes current state as input,
and output the next probable state

For example:
    - input: hi, how, are, you, doing
    
"""
from nltk import ngrams, word_tokenize
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import FloatTensor, randn, zeros

class RNN(nn.Module):
    """
    simple RNN network with 1 layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.h2h = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        # hidden_2 = self.h2h(hidden)
        # combined_2 = torch.cat((input, hidden_2), 1)
        output = self.i2o(combined)
        return output, hidden
    
    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.lstm_layers = nn.LSTM(input_size = input_size + hidden_size, 
                            hidden_size = hidden_size, 
                            num_layers = 1,
                            dropout = 0.1)

        self.fully_connected = nn.Linear(input_size + hidden_size, output_size)
        
        # self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        output, state = self.lstm(combined, hidden)
        logits = self.fully_connected(output)
    
    def init_hidden(self):
        return (Variable(torch.zeros(1, self.hidden_size)), 
                Variable(torch.zeros(1, self.hidden_size)))


