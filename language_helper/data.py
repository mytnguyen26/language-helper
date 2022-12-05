"""
This class load data from the local directory and return dataframe presentation
of that data
"""

import os
from typing import List
import pandas as pd
class Processing:
    def __init__(self, data_domain: str, data_dir: str = "../dataset/", delimiter: str = ","):
        path = os.path.join(data_dir,data_domain)
        self.delimiter = delimiter
        self.file_path: List[str] = []
        for file in os.listdir(path):
            if file.endswith(("txt", "csv")):
                self.file_path.append(os.path.join(path,file))
    

    def load(self):
        dataframe = pd.DataFrame()
        for file in self.file_path:
            with open(file, "r") as file:
                dataframe_stg = pd.read_csv(file, delimiter=self.delimiter)
                dataframe = pd.concat([dataframe, dataframe_stg])
        