import random
import json

import torch
import pandas as pd

from model.model import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize
from utils.dialogue_manager import DialogueManager


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_response(msg: str, msg_manager):

    _response = msg_manager.generate_answer(msg)
    
    return _response


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    df = pd.read_csv("data_corpus.csv")
    msg_manager = DialogueManager()

    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = _get_response(sentence,msg_manager)
        print(resp)

