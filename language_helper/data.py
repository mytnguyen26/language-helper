"""
This class load data from the local directory and return dataframe presentation
of that data
"""

import os
import re
import random
from typing import List
import pandas as pd
from pandas import DataFrame
import nltk
import torch
from nltk import word_tokenize
from typing import List, Dict, Any, Tuple

class Processing:
        
    def __init__(self, data_domain: str, data_dir: str = "./dataset/", delimiter: str = ","):
        nltk.download('punkt')
        path: str = os.path.join(data_dir,data_domain)
        self.delimiter: str = delimiter
        self.file_path: List[str] = []
        for file in os.listdir(path):
            if file.endswith(("txt", "csv")):
                self.file_path.append(os.path.join(path,file))
        self.list_of_tokens: List[List[str]] = None
        self.vocabulary_mapping: Dict[int, str] = None        

    def _create_vocab_mapping(self, list_of_tokens) -> Tuple[Dict[str,str], Dict[str, int]]:
        """
        from input sequences of tokens for each string, add to dictionary
        Args:
            - list_of_tokens List[List[str]]
        
        Returns:
            - vocabulary mapping Dict[str, str]: is the mapping of each unique word in
            the training set to a unique integer index
            For example: {0: "hello", 1: "hi", 2: "how", 3: "are", 4: "you"}

        """
        vocabulary_mapping = {}
        list_of_vocab_freq = {}
        index_cnt = 0

        for words in list_of_tokens:
            for token in words:
                try:
                    list_of_vocab_freq[token] += 1
                except Exception as error:
                    list_of_vocab_freq[token] = 1

        # creating vocab
        randomized_list_of_vocab = list(list_of_vocab_freq.keys())
        random.shuffle(randomized_list_of_vocab)
        
        for word in randomized_list_of_vocab:
            vocabulary_mapping[index_cnt] = word
            index_cnt += 1
        return vocabulary_mapping, list_of_vocab_freq

    def _create_tokens(self, dataframe, column) -> List[List[str]]:
        """
        From column in a dataframe, process each input string
        and return a list of tokens
        """
        dataframe_processing = dataframe[column]
        dataframe_processing = dataframe_processing.drop_duplicates()
        list_of_tokens = []

        for row in dataframe_processing:
            words = word_tokenize(row)
            words.append("</s>")
            list_of_tokens.append(words)

        # randomize input sentences for training
        random.shuffle(list_of_tokens)
        return list_of_tokens
        
    def load(self):
        dataframe = pd.DataFrame()
        for file in self.file_path:
            with open(file, "r") as file:
                print(f"Loading...{file}")
                dataframe_stg = pd.read_csv(file, delimiter=self.delimiter)

                print(f"{dataframe_stg.shape[0]} rows and {dataframe_stg.shape[1]} columns appended")
                dataframe = pd.concat([dataframe, dataframe_stg])

        print(f"Completed: {dataframe.shape[0]} rows and {dataframe.shape[1]} columns loaded")
        print(f"Columns {dataframe.columns}")
        return dataframe


    def prepare(self, dataframe: DataFrame, column: str) -> None:
        """
        Turn dataframe column into List of tokens
        Args:
            - dataframe:
            - column:
        Return:
            - list_of_tokens List[str]: a list of token with end of string padding </s>
            For example: ["]
        TODO
        """
        column = column.lower()
        dataframe.columns = [col.lower() for col in dataframe.columns]
        # lower case all values
        dataframe[column] = dataframe[column].apply(lambda x: str(x).lower())

        # remove punctuation
        dataframe[column] = dataframe[column].apply(lambda x: re.sub("\!|\?|\|\(|\)|,|\(|\)", "", x))
        dataframe[column] = dataframe[column].apply(lambda x: re.sub("-rrb-", "", x))
        dataframe[column] = dataframe[column].apply(lambda x: re.sub("-lrb-", "", x))
        dataframe[column] = dataframe[column].apply(lambda x: re.sub("[^a-zA-z0-9\s]", "", x))

        # save list of tokens to the instance
        self.list_of_tokens = self._create_tokens(dataframe, column)
        self.vocabulary_mapping, _ = self._create_vocab_mapping(self.list_of_tokens)
        

