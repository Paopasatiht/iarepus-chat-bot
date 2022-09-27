from distutils.log import debug
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import _get_response
from utils.dialogue_manager import DialogueManager
from sentence_transformers import SentenceTransformer
from utils.yamlparser import YamlParser
from models.model import NeuralNet
from pythainlp.corpus.common import thai_words
from pythainlp.util import Trie

import pandas as pd
import torch
import json


# Declare a Flask app :
app = Flask(__name__)
CORS(app)

#Declare a config file
config_file = "/Projects/configs/config.yaml"
cfg = YamlParser(config_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_corpus = pd.read_csv(cfg["DATA_CORPUS"]["data_csv"])

# Declare a custom dictionary :
custom_ls = cfg["CUSTOM_DICT"]["words"]
_dict = {k for k in custom_ls}
custom_words = _dict.union(thai_words())
custom_dictionary_trie = Trie(custom_words)

# Load sentence embedded model
answer_model = SentenceTransformer(cfg["MODEL"]["answer_model"])

# Load intent classfication model
with open('/Projects/configs/intents.json', 'r') as f:
    intents = json.load(f)

intent_path = cfg["MODEL"]["intent_model"]
data = torch.load(intent_path)

# Argument declaration for intent model
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

intent_model = NeuralNet(input_size, hidden_size, output_size).to(device)
intent_model.load_state_dict(model_state)
intent_model.eval()

# Main function here :
@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    msg_manager = DialogueManager(data_corpus, custom_dictionary_trie, answer_model, intent_model, input_size, hidden_size, output_size, all_words, tags, device)
    response = _get_response(text, msg_manager)
    message =  {"answer" : response}
    return jsonify(message)

# Runing the app :
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)