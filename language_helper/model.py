
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

class MarkovNgrams:
    def __init__(self, state):
        self.hash_map = {}
        
    # for each sentence in sentences
    # generate ngrams

    def matching_set(ngrams_model, tokens):
        """
        TODO
        """
        match_grams = []
        count = 0
        for grams in ngrams_model:
            print(f"compare {tokens} with {grams[:len(tokens)]}")
            if grams[:len(tokens)] == tokens:
                match_grams.append(grams)
                count += 1
        return match_grams

    def markov_next_word(self, match_grams):
        """
        TODO
        """
        chosen_word = ""
        p_chosen_word = -1
        # P(next_word | prev_word)
        for candidate in match_grams:
            next_word = candidate[-1]
            if next_word not in self.hash_map.keys():
                self.hash_map[next_word] = (1, 1/len(match_grams))
            else:
                self.hash_map[next_word][0] += 1
                self.hash_map[next_word][1] = self.hash_map[next_word][0]/len(match_grams)
            if self.hash_map[next_word][1] > p_chosen_word:
                chosen_word = next_word
                p_chosen_word = self.hash_map[next_word][1]
            elif self.hash_map[next_word][1] > p_chosen_word:
                np.random.choice([chosen_word, next_word])
            print(next_word)
        return (chosen_word, p_chosen_word)

    def make_model(batch_sentences: List[str], n: int):
        # making models
        model = []
        for sentence in batch_sentences:
            n_grams = ngrams(word_tokenize(sentence), n)
            for grams in n_grams:
                model.append(grams)
                print(f"last word should be the next word used for prediction, or target: {grams[n-1]}")
                print(f"prev words {grams[:n-1]}")
        return model

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


