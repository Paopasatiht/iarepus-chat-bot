from pythainlp import sent_tokenize, word_tokenize
from pythainlp import word_tokenize 
from pythainlp import word_vector
from sklearn.metrics.pairwise import cosine_similarity  
from pythainlp.word_vector import WordVector
import numpy as np


def thai_tokenize(sentence):
    return word_tokenize(
                        sentence,
                        keep_whitespace=False
    )

def thai_bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [word for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag

def sentence_vectorizer(ss, model, dim=300,use_mean=True):
    """ Receive a "sentence" and encode to vector in dimension 300
    Step : 
        1.) Word tokenize from "sentence"
        2.) Create a vector size == dimension
        3.) Add up the vector from the dictionary of index2word
        4.) return sentence vectorize
    """
    s = word_tokenize(ss)
    vec = np.zeros((1,dim))
    for word in s:
        if word in model.index_to_key:
            vec+= model.get_vector(word)
        else: pass
    if use_mean: vec /= len(s)
    return vec

def sentence_similarity(s1,s2,model=None):
    """ Measure the simirality from "two sentence"
    """
    return cosine_similarity(sentence_vectorizer(str(s1), model),sentence_vectorizer(str(s2), model))

def _word_tkn(s1 : str):

    return word_tokenize(s1)