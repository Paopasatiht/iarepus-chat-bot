import random
import json

import torch
import pandas as pd

from models.model import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize
from utils.dialogue_manager import DialogueManager
from sentence_transformers import SentenceTransformer
from utils.yamlparser import YamlParser


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_file = "/Projects/configs/config.yaml"
cfg = YamlParser(config_file)

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

def _get_response(msg: str, msg_manager):

    _response = msg_manager.generate_answer(msg)
    
    return _response


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    msg_manager = DialogueManager(answer_model, intent_model, input_size, hidden_size, output_size, all_words, tags, device)

    while True:
        try:
            sentence = input("You: ")
            if sentence == "quit":
                break

            resp = _get_response(sentence,msg_manager)
            print(resp)
        except Exception as e:
            print("Error!, Do not use backspace at the end of line")
            
