import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class IntentsClassification():

    def __init__(self, intent_model, prob_model, count_vec):

        self.intent_model = intent_model
        self.prob_model = prob_model
        self.count_vec = count_vec

    def predict_class(self, sentence):

        input_vec = self.count_vec.transform([sentence])
        tag = self.intent_model.predict(input_vec)
        return tag[0]



    def predict_score(self, sentence):

        input_vec = self.count_vec.transform([sentence])
        predicted = self.prob_model.predict_proba(input_vec)
        max_ind = np.argmax(predicted, axis=1)
        pred = predicted[0][max_ind]
        return pred

    def predict_tagging(self, sentence):

        tag_dict ={}
        tag_dict.update({self.predict_class(sentence) : self.predict_score(sentence)})

        return tag_dict

