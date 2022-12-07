import pytest
from language_helper.markov.model import MarkovNgrams

@pytest.fixture
def markov():
    markov = MarkovNgrams()
    markov.ngrams_model = [('how', 'are', 'you'),
                        ('are', 'you', 'doing'),
                        ('you', 'doing', '?'),
                        ('i', "'m", 'fine'),
                        ("'m", 'fine', '.'),
                        ('i', "'m", 'pretty'),
                        ("'m", 'pretty', 'good'),
                        ('pretty', 'good', '.'),
                        ('no', 'problem', '.'),
                        ("'m", 'pretty', 'well'),
                        ("he", 'pretty', 'well')]
    return markov

def test_make_models(markov):
    """
    From a list of input string
    generate n_grams model
    """
    input = [["hello", "how", "are", "you"],
            ["Im", "good", "how", "are", "you"]]

    markov.make_model(input, 3)
    assert len(markov.ngrams_model) == 5


def test_predict_next_word_case_a(markov):
    input_str = "how are you"
    chosen_word, p = markov.predict_next_word(input_str)
    assert chosen_word == "doing"


def test_predict_next_word_case_b(markov):
    """
    test when there are multiple matches
    calculate occurences and yield most frequent
    """
    input_str = "pretty"
    chosen_word, p = markov.predict_next_word(input_str)
    assert chosen_word == "well"

def test_predict_next_word_out_of_vocab(markov):
    pass

def test_matching_set_case_a(markov):
    input_token = ["how", "are"]
    match_grams = markov.matching_set(tokens=input_token)
    assert match_grams == [('how', 'are', 'you')]
    
def test_matching_set_case_b(markov):
    input_token = ["are"]
    match_grams = markov.matching_set(tokens=input_token)
    assert match_grams == [('how', 'are', 'you')]

def test_matching_set_case_c(markov):
    input_token = ["hi", "how", "are"]
    match_grams = markov.matching_set(tokens=input_token)
    assert match_grams == [('how', 'are', 'you')]
