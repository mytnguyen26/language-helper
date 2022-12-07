
import numpy as np
from nltk import ngrams, word_tokenize
from typing import List, Tuple

class MarkovNgrams:
    def __init__(self):

        self.result_hash_map = {}
        self.ngrams_model: List[List[str]] = None

    def matching_set(self, tokens: List[str]) -> List[Tuple]:
        """
        Perform a search in class instance ngrams_model for
        matching entries against tokens input, and return a list
        of matching entries against token.
        Token size could be equal, greater, or less than the size of
        each entry in ngrams_model. The comparison will only compare upto
        the size of each entry - 1
        (refer to tests/test_model_markov for detailed tests)

        Args:
            - tokens (List[str]): is the tokenized input string. such as
            ["how", "are", "you"]

        Return:
            - match_grams (List[Tuple]): is the list of matching entries
        """
        match_grams = []
        count = 0
        for grams in self.ngrams_model:
            if len(tokens) <= len(grams) - 1:
                if grams[-len(tokens)-1:-1] == tuple(tokens):
                    match_grams.append(grams)
                    count += 1
            else:
                # match the last index of grams
                if grams[:-1] == tuple(tokens[-(len(grams)-1):]):
                    match_grams.append(grams)
                    count += 1

        return match_grams
    
    def make_model(self, batch_sentences: List[List[str]], n_grams: int) -> List[List[str]]:
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
            list_ngrams = ngrams(sentence, n_grams)
            for grams in list_ngrams:
                ngrams_model.append(grams)
                print(f"last word should be the next word used for prediction, or target: {grams[n_grams-1]}")
                print(f"prev words {grams[:n_grams-1]}")
        self.ngrams_model = ngrams_model

    def feedback(self, input):
        """
        When the model encounter new sequences (i.e. predict next word returns ''),
        this phenomenon known as Out of Vocabulary. Use input provided by the human
        and iteratively add to model
        """
        pass

    def predict_next_word(self, input: str):
        """
        From input, find the next best word based on the likelihood (frequency) 
        of that word given that `input_token` is the preceeding word.

        For example, if the ngrams_model contains the following entries:
            [
                ("hi", "how", "are"),
                ("how", "are", "you")
                ("hi", "how", "are"),
                ("how", "are", "your"],
                ("are", "your", "dog")
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
        result_hash_map = {}
        input_token = word_tokenize(input)
        
        match_grams = self.matching_set(input_token)
        chosen_word = ""
        p_chosen_word = -1
        # P(next_word | prev_word)

        for candidate in match_grams:
            next_word = candidate[-1]
            if next_word not in result_hash_map.keys():
                result_hash_map[next_word] = [1, 1/len(match_grams)]
                
            else:
                result_hash_map[next_word][0] += 1
                result_hash_map[next_word][1] = result_hash_map[next_word][0]/len(match_grams)
            if result_hash_map[next_word][1] > p_chosen_word:
                chosen_word = next_word
                p_chosen_word = result_hash_map[next_word][1]
            elif result_hash_map[next_word][1] == p_chosen_word:
                np.random.choice([chosen_word, next_word])

        self.result_hash_map = result_hash_map

        return (chosen_word, p_chosen_word)

