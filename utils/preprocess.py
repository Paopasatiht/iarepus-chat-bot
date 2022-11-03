from typing import List
import pandas as pd
import re
import string
from pythainlp.util import normalize
from pythainlp.spell import spell
from pythainlp import word_tokenize

from pythainlp.corpus import thai_stopwords
from pythainlp import correct



def clean_msg(msg : str):
    """ Preprocess by remove the special character and remove white space from the "sentence"
    """

    msg = re.sub(r'<.*?>','', msg)    
    msg = re.sub(r'#','',msg)
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c),'',msg)
    msg = ''.join(msg.split())
    
    return msg

def normalize_msg(msg : str):
    
    return normalize(msg)

def spell_corrector(msg : str):
    """ Receive a word and process the spell checker
    """
    # r_msg = correct(msg)
    word_list = word_tokenize(msg, engine="longest",keep_whitespace=False)
    print("ตัดคำ : {}".format(word_list))
    r_msg = ''.join(correct(w) for w in word_list)

    return  r_msg

def remove_stop_words(list_word):
    stopwords = list(thai_stopwords())
    list_word_not_stopwords = [i for i in list_word if i not in stopwords]


def preprocess_text(sentence : str):

    """ To preprocess the user input text:
    1.) Remove all special punctuation
    2.) normalize the text
    3.) Correct misspelling
    """

    remove_p_txt = clean_msg(sentence)
    normalize_text = normalize_msg(remove_p_txt)
    # clean_txt = spell_corrector(normalize_text)

    return normalize_text

def generate_n_gram(sentence : str, n = 5) -> List:
    
    tokens = [token for token in word_tokenize(sentence) if token != ""]
    
    if len(tokens) < n:
        return tokens
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]
    
def get_th_tokens(text : str):

  text = text.lower()
  text = text.replace('\n', ' ')
  tokens = word_tokenize(text,keep_whitespace=False)
  
  return tokens