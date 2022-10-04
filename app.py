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

# Declare a Flask app :
app = Flask(__name__)
CORS(app)

#Declare a config file
config_file = "/Projects/configs/config.yaml"
cfg = YamlParser(config_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_corpus = pd.read_csv(cfg["DATA_CORPUS"]["data_csv"])

# Load sentence embedded model
answer_model = SentenceTransformer(cfg["MODEL"]["answer_model"])

# Load intent classification model
intent_path = cfg["MODEL"]["intent_model"]
prob_path = cfg["MODEL"]["prob_model"]

# Load model from weight files
intent_model = pickle.load(open(intent_path, 'rb'))
prob_model = pickle.load(open(prob_path, 'rb'))


# Declare count vectorizer
tf_vectorizer = CountVectorizer()
vectors = tf_vectorizer.fit_transform(data_corpus.Keys)

# Load sentence embedded model
answer_model = SentenceTransformer(cfg["MODEL"]["answer_model"])
wv = WordVector()
wv_model = wv.get_model()

# tags declaration
kw = list(cfg["KEYWORD_INTENT"].keys())

# Main function here :
@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    msg_manager = DialogueManager(data_corpus, wv_model, answer_model, intent_model, prob_model, tf_vectorizer, device, kw)
    response = _get_response(text, msg_manager)
    message =  {"answer" : response}
    return jsonify(message)

# Runing the app :
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)