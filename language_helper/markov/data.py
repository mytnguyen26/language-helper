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
from nltk import word_tokenize, sent_tokenize
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

    def _create_tokens(self, dataframe, column) -> List[List[str]]:
        """
        From column in a dataframe, process each input string
        and return a list of tokens
        For example, if the input string is "hello how are you?"
        then lists_of_tokens = ["hello", "how", "are", "you", "</s>"]
        """
        dataframe_processing = dataframe[column]
        list_of_tokens = []

        for row in dataframe_processing:
            # sentence = sent_tokenize(row)
            words = word_tokenize(row)
            words.append("</s>")
            list_of_tokens.append(words)

        # randomize input sentences for training
        random.shuffle(list_of_tokens)
        return list_of_tokens
        
    def load(self):
        dataframe = pd.DataFrame()

        for file in self.file_path:
            with open(file, mode="r") as file:
                rows = file.readlines()
                rows = [row.replace("\t"," ").replace("\n", "") for row in rows]
            
            print(f"Loading...{file}")
            dataframe_stg = pd.DataFrame(rows, columns=["data"])

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
            For example: ["hello", "how", "are", "you", "</s>"]
        """
        column = column.lower()
        dataframe.columns = [col.lower() for col in dataframe.columns]
        # lower case all values
        dataframe[column] = dataframe[column].apply(lambda x: str(x).lower())

        # remove punctuation
        dataframe[column] = dataframe[column].apply(lambda x: re.sub("\.|\!\?\|\(|\)|,|\(|\)", "", x))
        dataframe[column] = dataframe[column].apply(lambda x: re.sub("-rrb-", "", x))
        dataframe[column] = dataframe[column].apply(lambda x: re.sub("-lrb-", "", x))
        dataframe[column] = dataframe[column].apply(lambda x: re.sub("[^a-zA-z0-9\s]", "", x))

        # save list of tokens to the instance
        self.list_of_tokens = self._create_tokens(dataframe, column)


