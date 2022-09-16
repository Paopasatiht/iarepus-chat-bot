import random
import json

import torch
import pandas as pd

from models.model import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize
from utils.dialogue_manager import DialogueManager
from sentence_transformers import SentenceTransformer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =  SentenceTransformer('mrp/simcse-model-roberta-base-thai')


def _get_response(msg: str, msg_manager):

    _response = msg_manager.generate_answer(msg)
    
    return _response


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    df = pd.read_csv("../Projects/data_corpus.csv")
    msg_manager = DialogueManager(model)

    while True:
        try:
            sentence = input("You: ")
            if sentence == "quit":
                break

            resp = _get_response(sentence,msg_manager)
            print(resp)
        except:
            print("Error!, Do not use backspace at the end of line")

