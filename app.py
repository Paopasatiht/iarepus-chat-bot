from distutils.log import debug
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import _get_response
from utils.dialogue_manager import DialogueManager
from sentence_transformers import SentenceTransformer
from utils.yamlparser import YamlParser
from models.model import NeuralNet
from pythainlp.corpus.common import thai_words
from pythainlp.word_vector import WordVector
from pythainlp.util import Trie

import pandas as pd

import torch
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from utils.helper import get_th_tokens

# from pythainlp.word_vector import WordVector
from gensim.models import KeyedVectors

# Declare a Flask app :
app = Flask(__name__)
CORS(app)

#Declare a config file
config_file = "/Projects/configs/config.yaml"
cfg = YamlParser(config_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data_corpus = pd.read_csv(cfg["DATA_CORPUS"]["data_csv"])
keyword_csv = pd.read_csv(cfg["DATA_CORPUS"]["keyword_csv"])
print("Prepare data corpus")

# Load sentence embedded model
answer_model = SentenceTransformer(cfg["MODEL"]["answer_model"])
print("Load sentence embedded model . . .")

# Load intent classification model
intent_path = cfg["MODEL"]["intent_model"]

# Load model from weight files
intent_model = pickle.load(open(intent_path, 'rb'))
print("Load intent model . . .")

# Declare count vectorizer
tf_vectorizer = CountVectorizer(tokenizer=get_th_tokens, ngram_range = (1, 2))
vectors = tf_vectorizer.fit_transform(data_corpus.Keys)

# Load sentence embedded model
answer_model = SentenceTransformer(cfg["MODEL"]["answer_model"])
wv_model = KeyedVectors.load_word2vec_format('/Projects/checkpoints/LTW2V_v0.1.bin', binary=True, unicode_errors='ignore')
print("Load word vector model . . .")

 # tags declaration
kw = list(cfg["KEYWORD_INTENT"].keys())

# Declared a dictionary from config.yaml files
config_dict = cfg["KEYWORD_INTENT"]

# Declare a custom dictionary :
print("Prepare custom dictionary ...")
custom_ls = cfg["CUSTOM_DICT"]["words"]
_dict = {k for k in custom_ls}
custom_words = _dict.union(thai_words())
custom_dictionary_trie = Trie(custom_words)

# Main function here :
@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    msg_manager = DialogueManager(data_corpus, wv_model, answer_model, intent_model, tf_vectorizer, config_dict, custom_dictionary_trie, keyword_csv, kw)
    response = _get_response(text, msg_manager)

    message =  {"answer" : response}

    return jsonify(message)

# Runing the app :
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=False)