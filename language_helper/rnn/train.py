from typing import List
from torch import save, load
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch import FloatTensor, randn, zeros
from torch.utils.data import random_split
import random

class Training:
    def __init__(self, model, criterion, encode, \
                n_iterations: int, learning_rate: float, optimizer = None, training_set = None):
        
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.n_iterations = n_iterations
        self.training_result: List[float] = []
        self.training_set: List[List[str]] = training_set
        self.validation_set: List[List[str]] = None
        self.encode = encode

    def _train_each_sentence(self, iter):
        """
        TODO
        """
        hidden = self.model.init_hidden()
        total_loss = 0

        sentence = random.choice(self.training_set)

        # wrap tensor in Variable
        x_tensor, y_tensor = self.encode.formulate_target(sentence)
        x_tensor_wrap = Variable(x_tensor)
        y_tensor_wrap = Variable(y_tensor)

        # iterate thru token in each sentence
        for index in range(x_tensor_wrap.size()[0]):
            # zero gradient
            if self.optimizer != None:
                self.optimizer.zero_grad()
            else:
                self.model.zero_grad()
                
            # optimizer.zero_grad()
            output, hidden = self.model(x_tensor_wrap[index], hidden)

            # to evaluate output, we need to compare generated output
            # against the actual next word
        
            # after training each word in input sequence, calculate loss
            loss = self.criterion(output, y_tensor_wrap[index])

            # recording the loss for total loss of the sentence
            total_loss += loss.item()
            
            # https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/2
            # To reduce memory usage, during the .backward() call, all the intermediary results
            # are deleted when they are not needed anymore. Hence if you try to call .backward() again,
            # the intermediary results don’t exist and the backward pass cannot be performed (and you get the error you see).
            # You can call .backward(retain_graph=True) to make a backward pass that will not delete intermediary results
            loss.backward(retain_graph=True)
            
            # backpropagate
            if self.optimizer != None:
                self.optimizer.step()
            else:
                
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            # update weights
            for p in self.model.parameters():
                try:
                    p.data.sub_(-self.learning_rate, p.grad.data)
                except Exception as e:
                    print("Skipping")
        
        return output, total_loss/x_tensor.size()[0]

    def split_train_test(self, training_size):
        self.training_set, self.validation_set = random_split(self.training_set, training_size)

    def early_stopping(self):
        """
        Ending model training and output to a pickle file if loss
        stop reducing
        """
        pass

    def save_checkpoint(self, iter, loss):
        """
        Add checkpoint for the model. Output model to a pickle file
        """
        if self.optimizer != None:
            optimizer_state_dict = self.optimizer.state_dict()
        save({
            'epoch': iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer_state_dict,
            'loss': loss,
        }, "./model_checkpoints/model.pt")

    def load_checkpoint(self):
        """
        Load a model checkpoint and update model state
        with state dict saved in checkpoint
        """
        checkpoint = load("./model_checkpoints/model.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer != None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            iter = checkpoint['epoch']
            loss = checkpoint['loss']
        return iter, loss
    
    def run(self):
        """
        TODO
        """
        self.model.train()
        for iter in range(1, self.n_iterations+1):
            output, avg_sentence_loss = self._train_each_sentence(iter)
            self.training_result.append(float(avg_sentence_loss))