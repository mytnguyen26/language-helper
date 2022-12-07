import nltk
import torch
from torch.autograd import Variable


class PredictRNN:

    @staticmethod
    def predict(input_sentence, rnn, encode, hidden_size, vocabulary_mapping, output_length=None):
        """
        TODO
        """
        input_sentence = nltk.word_tokenize(input_sentence)
        
        predicted_str = []
        rnn.eval()
        hidden = rnn.init_hidden()
        for i in range(output_length+1):  
            
            x_input = encode.sentence_to_tensor(input_sentence, len(vocabulary_mapping))
            x_input = Variable(x_input)
            
            hidden = Variable(torch.zeros(1, hidden_size))
            output, next_hidden = rnn(x_input[0], hidden)

            # get the token probabilities
            p = torch.nn.functional.softmax(output, dim=1).data

            # use top 1 word as next input
            input_sentence = [vocabulary_mapping[int(output.data.topk(1).indices[0][0])]]
            # print(f"next input {input_sentence}")
            predicted_str.append(input_sentence[0])
        return predicted_str

class PredictLSTM:

    @staticmethod
    def predict(input_sentence, rnn, encode, hidden_size, vocabulary_mapping, output_length=None):
        """
        TODO
        """
        input_sentence = nltk.word_tokenize(input_sentence)
        
        predicted_str = []
        rnn.eval()
        hidden_init, cell_init = rnn.init_hidden()
        for i in range(output_length+1):  
            
            x_input = encode.sentence_to_tensor(input_sentence, len(vocabulary_mapping))
            x_input = Variable(x_input)
            
            output, (next_hidden, cell) = rnn(x_input[0], hidden_init, cell_init)

            # get the token probabilities
            p = torch.nn.functional.softmax(output, dim=1).data

            # use top 1 word as next input
            input_sentence = [vocabulary_mapping[int(output.data.topk(1).indices[0][0])]]
            
            # print(f"next input {input_sentence}")
            predicted_str.append(input_sentence[0])
        return predicted_str
