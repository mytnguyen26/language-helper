
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
    def __init__(self, input_size, hidden_size, layer_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.output_size = output_size

        # define LSTM layers using pytorch LSTM module
        self.lstm_layers = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size, 
                            num_layers = layer_size,
                            batch_first=False,
                            dropout = 0.1)

        self.fully_connected = nn.Linear(hidden_size, output_size)
        
        # self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, cell):
        
        output, (hidden_state, cell_state) = self.lstm_layers(input, (hidden, cell))
        logits = self.fully_connected(output)

        return logits, (hidden_state, cell_state)
    
    def init_hidden(self):
        """
        From pytorch documentation:
        initial hidden state has tensor of shape (D * num_layers}, hidden_size) for
        unbatched input
        if bidirectional, D = 2. Otherwise, D = 1
        """
        hidden_state = Variable(torch.zeros(self.layer_size, self.hidden_size)).requires_grad_()
        cell_state = Variable(torch.zeros(self.layer_size, self.hidden_size)).requires_grad_()
        return (hidden_state, cell_state)
                


