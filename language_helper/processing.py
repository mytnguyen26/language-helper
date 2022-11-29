from typing import Tuple, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import FloatTensor, randn, zeros
from model import MarkovNgrams, RNN

class ProcessingRNN:
    def __init__(self, input_path, model, criterion, **kwargs):
        self.vocabulary_mapping: Dict[int, str] = self._create_vocabulary_mapping()
        self.input_path: str = input_path
        self.model = model
        self.criterion = criterion
        self.learning_rate = kwargs["learning_rate"]
        self.input_size = kwargs["input_size"]
        self.hidden_size = kwargs["hidden_size"]
        self.output_size = kwargs["output_size"]
        self.n_iterations = kwargs["n_iterations"]

    def _create_vocabulary_mapping(self):
        """
        From Dataframe, generate a mapping or word to integer
        """
        
        vocabulary_mapping = {}
        list_of_vocab = []
        sentences = []
        index_cnt = 0

        for row in dataframe.iterrows():
            words = word_tokenize(row[1][column])
            sentences.append(words)
            list_of_vocab.extend(words)

        unique_list_of_vocab = np.unique(list_of_vocab)

        for word in unique_list_of_vocab:
            vocabulary_mapping[index_cnt] = word
            index_cnt += 1
        vocabulary_mapping[index_cnt] = "</s>"
        return vocabulary_mapping, sentences

    def _find_word_index(self, word):
        """
        Look up word index in vocabulary dictionary
        """
        value = [index for index in self.vocabulary_mapping \
                    if self.vocabulary_mapping[index] == word]
        return value[0]

    def _sentence_to_tensor(self, sentences):
        tensor = torch.zeros(len(sentences), 1, len(self.vocabulary_mapping))
        for sentence, word in enumerate(sentences):
            index = self.find_word_index(word)
            tensor[sentence][0][index] = 1
        return tensor

    def encode(self, sentences) -> Tuple[Any, Any]:
        """
        TODO
        """
        # adding padding end of sentences
        orig_size = len(sentences)
        x_sentence = sentences
        y_sentence = sentences + ["</s>"]
        x_tensor = self._sentence_to_tensor(sentences=x_sentence)
        y_tensor = self._sentence_to_tensor(sentences=y_sentence[1:])
        return x_tensor, y_tensor
        
    def train(self, input_sequence):
        """
        TODO
        """

        hidden = self.model.init_hidden()

        # select an input for training
        loss = 0
        sentence = np.random.choice(sentences)
        
        # wrap tensor in Variable
        x_tensor, y_tensor = formulate_target(sentence)
        x_tensor = Variable(x_tensor)
        y_tensor = Variable(y_tensor)

        # zero gradient
        rnn.zero_grad()

        for index in range(x_tensor.size()[0]):
            
            output, hidden = rnn(x_tensor[index], hidden)

            # to evaluate output, we need to compare generated output
            # against the actual next word
        
            # after training each word in input sequence, calculate loss
            loss += criterion(output, y_tensor[index])

        # https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/2
        # To reduce memory usage, during the .backward() call, all the intermediary results
        # are deleted when they are not needed anymore. Hence if you try to call .backward() again,
        # the intermediary results don’t exist and the backward pass cannot be performed (and you get the error you see).
        # You can call .backward(retain_graph=True) to make a backward pass that will not delete intermediary results
        
        loss.backward(retain_graph=True)

        # update weights
        for p in rnn.parameters():
            p.data.add_(-LEARNING_RATE, p.grad.data)


    def predict():
        pass