import random
import json

import torch
import pickle
import pandas as pd

from models.model import NeuralNet
from utils.dialogue_manager import DialogueManager
from sentence_transformers import SentenceTransformer
from utils.yamlparser import YamlParser
from pythainlp.corpus.common import thai_words
from pythainlp.util import Trie
from sklearn.feature_extraction.text import CountVectorizer

from pythainlp.word_vector import WordVector

    
def _get_response(msg: str, msg_manager):

    _response = msg_manager.generate_answer(msg)
    
    return _response


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_file = "/Projects/configs/config.yaml"
    cfg = YamlParser(config_file)
    data_corpus = pd.read_csv(cfg["DATA_CORPUS"]["data_csv"])

    # Load sentence embedded model
    answer_model = SentenceTransformer(cfg["MODEL"]["answer_model"])
    wv = WordVector()
    wv_model = wv.get_model()

    # Declare a custom dictionary :
    custom_ls = cfg["CUSTOM_DICT"]["words"]
    _dict = {k for k in custom_ls}
    custom_words = _dict.union(thai_words())
    custom_dictionary_trie = Trie(custom_words)

    # Load intent classfication model
    with open('/Projects/configs/intents.json', 'r') as f:
        intents = json.load(f)

    intent_path = cfg["MODEL"]["intent_model"]
    prob_path = cfg["MODEL"]["prob_model"]

    intent_model = pickle.load(open(intent_path, 'rb'))
    prob_model = pickle.load(open(prob_path, 'rb'))

    tf_vectorizer = CountVectorizer()
    vectors = tf_vectorizer.fit_transform(data_corpus.Keys)

    # tags declaration
    kw = list(cfg["KEYWORD_INTENT"].keys())

    print("Let's chat! (type 'quit' to exit)")
    msg_manager = DialogueManager(data_corpus, wv_model, answer_model, intent_model, prob_model, tf_vectorizer, device, kw)

    while True:
        try:
            sentence = input("You: ")
            if sentence == "quit":
                break

            resp = _get_response(sentence,msg_manager)
            print(resp)
        except Exception as e:
            print(e)
            
