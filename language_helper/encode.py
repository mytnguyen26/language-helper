
from typing import List
import torch

class EncodeUtility:
    @staticmethod
    def sentence_to_tensor(self, tokens: List[str], vocab_size: int):
        """
        Create a tensor for each sentence.

        Args:
            - tokens: a sentence splited into token. 
            For example: ["hi", "how", "are", "you", "</s>"]
        Return:
            - tensor:
        """
        tensor = torch.zeros(len(tokens), 1, len(vocab_size))
        for sentence, word in enumerate(tokens):
            index = self._find_word_index(word)
            tensor[sentence][0][index] = 1
        return tensor

    @staticmethod
    def formulate_target(self, sentences: List[str]):
        """
        For example:
        string: This is a test

        embedded tensor:
        This = [[[]]]

        target: is
        """
        orig_size = len(sentences)
        x_sentence = sentences[:-1]
        y_sentence = sentences[1:]
        x_tensor = self._sentence_to_tensor(sentences=x_sentence)
        y_tensor = self._sentence_to_tensor(sentences=y_sentence)
        return x_tensor, y_tensor

    @staticmethod
    def find_word_index(self, word, vocabulary_mapping):
        """
        Look up word index in vocabulary dictionary
        """
        value = [index for index in vocabulary_mapping.keys() if vocabulary_mapping[index] == word]
        return value[0]