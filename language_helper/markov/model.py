
import numpy as np
from nltk import ngrams, word_tokenize
from typing import List

class MarkovNgrams:
    def __init__(self, state):
        self.result_result_hash_map = {}
        self.ngrams_model: List[List[str]] = None
        
    # for each sentence in sentences
    # generate ngrams

    
    def matching_set(self, ngrams_model, tokens):
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
    
    def make_model(batch_sentences: List[str], n_grams: int) -> List[List[str]]:
        """
        Go thru set of sentences, break sentences into ngrams.
        Finally return a list of ngrams token.
        For example, if we have a sentence:
            
            ["hi", "how", "are", "you"]
        
        With n_grams = 3, we will generate lists of tokens, with each list
        has length 3. The final index token is the word we want to predict, given
        the last 2 previous words

            [
                ["hi", "how", "are"],
                ["how", "are", "you"]
            ]

        The first list ["hi", "how", "are"] try to represent P["are" | "how", "hi"].
        These lists of of tokens will be used to calculate the frequency of the last word,
        for each Markov state.

        ------------------------------------------------------
        Args:
            + batch_sentences: list of sentence we want to create ngrams from
            + n_grams (int): the number of ngrams to generate.

        Returns:
            + ngrams_model: lists of token. Each list has `n_grams` number of token
            For example: 
            [
                ["hi", "how", "are"],
                ["how", "are", "you"]
            ]
        """
        ngrams_model = []
        for sentence in batch_sentences:
            n_grams = ngrams(word_tokenize(sentence), n_grams)
            for grams in n_grams:
                ngrams_model.append(grams)
                print(f"last word should be the next word used for prediction, or target: {grams[n_grams-1]}")
                print(f"prev words {grams[:n_grams-1]}")
        return ngrams_model

    def predict(self, input: str):
        """
        From input token, find the next best word based on the likelihood (frequency) 
        of that word given that `input_token` is the preceeding word.

        For example, if the ngrams_model contains the following entries:
            [
                ["hi", "how", "are"],
                ["how", "are", "you"]
                ["hi", "how", "are"],
                ["how", "are", "your"],
                ["are", "your", "dog"]
            ]
        And input_tokens = "how are". Model finds entries with the first 2 tokens
        contains "how", "are". Then its counts the number of times the last word being "you", "your", etc.
        Finally, it select the word with the highest frequency.

        Args:
            - input (str): is the input string that we will use to predict the next word

        Returns:
            - chosen_word, p_chosen_word (Tuple[str, float]): the chosen word and its corresponding
            frequency

        """

        input_token = word_tokenize(input)
        
        match_grams = self.matching_set(self.ngrams_model, input_token)
        chosen_word = ""
        p_chosen_word = -1
        # P(next_word | prev_word)

        for candidate in match_grams:
            next_word = candidate[-1]
            if next_word not in self.result_hash_map.keys():
                self.result_hash_map[next_word] = (1, 1/len(match_grams))
            else:
                self.result_hash_map[next_word][0] += 1
                self.result_hash_map[next_word][1] = self.result_hash_map[next_word][0]/len(match_grams)
            if self.result_hash_map[next_word][1] > p_chosen_word:
                chosen_word = next_word
                p_chosen_word = self.result_hash_map[next_word][1]
            elif self.result_hash_map[next_word][1] > p_chosen_word:
                np.random.choice([chosen_word, next_word])
            print(next_word)
        return (chosen_word, p_chosen_word)