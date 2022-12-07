import pytest
import pandas as pd
from torch import equal
from language_helper.rnn.data import Processing
from language_helper.rnn.encode import EncodeUtility


@pytest.fixture
def processing():
    processor = Processing(data_domain="tests", data_dir=".", delimiter="\t")
    return processor

def test_create_vocab_mapping(processing):
    input_tokens = [["hello", "how", "are", "you"], 
                    ["hi", "how", "are", "you"]]

    vocabulary_mapping, _ = processing._create_vocab_mapping(input_tokens)
    assert len(vocabulary_mapping) == 5


def test_create_tokens(processing):
    input_tokens = ["hello how are you", 
                    "good how about you"]
    dataframe = pd.DataFrame(input_tokens, columns=["data"])

    list_of_tokens = processing._create_tokens(dataframe, "data")
    assert len(list_of_tokens) == 2

def test_sentence_to_tensor(processing):
    input_tokens = [["hello", "how", "are", "you"], 
                    ["hi", "how", "are", "you"]]

    vocabulary_mapping, _ = processing._create_vocab_mapping(input_tokens)
    tokens = ["how", "are", "you"]
    encoder = EncodeUtility(vocabulary_mapping=vocabulary_mapping)
    tensor = encoder.sentence_to_tensor(tokens=tokens, vocab_size=len(vocabulary_mapping))
    assert tensor.shape[0] == 3
    assert tensor.shape[1] == 1
    assert tensor.shape[2] == 5 

def test_formulate_target(processing):
    input_tokens = [["hello", "how", "are", "you"], 
                    ["hi", "how", "are", "you"]]

    vocabulary_mapping, _ = processing._create_vocab_mapping(input_tokens)
    tokens = ["how", "are", "you"]
    encoder = EncodeUtility(vocabulary_mapping=vocabulary_mapping)
    input_tensor, target_tensor = encoder.formulate_target(sentences=tokens)
    assert input_tensor.shape[0] == 2
    assert input_tensor.shape[1] == 1
    assert input_tensor.shape[2] == 5
    assert target_tensor.shape[0] == 2
    assert target_tensor.shape[1] == 1
    assert target_tensor.shape[2] == 5
    assert not equal(input_tensor[0][0], target_tensor[0][0])
    assert not equal(input_tensor[1][0], target_tensor[1][0])

