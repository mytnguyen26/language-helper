
from typing import List
import torch

class EncodeUtility:
    def __init__(self, vocabulary_mapping = None) -> None:
        self.vocabulary_mapping = vocabulary_mapping
    
    def sentence_to_tensor(self, tokens: List[List[str]], vocab_size: int):
        """
        Create a tensor for each sentence.

        Args:
            - tokens: a sentence splited into token. 
            For example: ["hi", "how", "are", "you", "</s>"]
        Return:
            - tensor: a 3-dimensions array-like object of size
            [length_of_sentence, 1, vocabulary_size]
            For example:
                [[[0, 1, 0]],
                [[1, 0, 1]]]
        """
        tensor = torch.zeros(len(tokens), 1, vocab_size)
        for sentence, word in enumerate(tokens):
            index = self.find_word_index(word)
            tensor[sentence][0][index] = 1
        return tensor

    
    def formulate_target(self, sentences: List[List[str]]):
        """
        Create input and target tensor for each list_of_tokens
        Args:
            - sentences (List[List[str]]): list of tokens extracted from
            a string sentence
            For example: [["hello", "how", "are", "you"]]

        Returns:
            - x_tensor: a tensor object of size (length of sentence -1 , 1, size of vocab)
            - y_tensor: a tensor object of size (length of sentence -1, 1, size of vocab)
        """
        orig_size = len(sentences)
        x_sentence = sentences[:-1]
        y_sentence = sentences[1:]
        x_tensor = self.sentence_to_tensor(tokens=x_sentence, vocab_size=len(self.vocabulary_mapping))
        y_tensor = self.sentence_to_tensor(tokens=y_sentence, vocab_size=len(self.vocabulary_mapping))
        return x_tensor, y_tensor

    
    def find_word_index(self, word):
        """
        Look up word index in vocabulary dictionary
        """
        value = [index for index in self.vocabulary_mapping.keys() if self.vocabulary_mapping[index] == word]
        return value[0]