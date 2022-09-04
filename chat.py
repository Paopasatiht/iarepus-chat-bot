import random
import json

import torch
import pandas as pd

from model.model import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize
from utils.dialogue_manager import DialogueManager


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# def get_response(msg):
#     sentence = tokenize(msg)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)
#
#     output = model(X)
#     _, predicted = torch.max(output, dim=1)
#
#
#     print("Check here {}".format(predicted.item()))
#     tag = tags[predicted.item()]
#
#
#     #TODO: Add this part for pattern matching
#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 return random.choice(intent['responses'])
#
#     return "I do not understand..."

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

