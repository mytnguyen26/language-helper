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
                        ('no', 'problem', '.')]
    return markov

def test_make_models(markov):
    pass


def test_predict_next_word(markov):
    input_str = "how are you"
    chosen_word, p = markov.predict_next_word(input_str)
    assert chosen_word == "doing"


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