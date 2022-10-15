import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pythainlp import word_tokenize
from utils.yamlparser import YamlParser

from utils.preprocess import generate_n_gram

config_file = "/Projects/configs/config.yaml"
cfg = YamlParser(config_file)
kw = cfg["KEYWORD_INTENT"]


class IntentsClassification():

    def __init__(self, word_vector_model, intent_model, count_vec, tags):

        self.intent_model = intent_model
        self.count_vec = count_vec
        self.confidence_score = 0.40
        self.wv_model = word_vector_model
        self.tags = tags
        self.weights_standout = 0.50

    def word_embedded(self ,model, sentence, dim = 400, use_mean = True):
        """ Receive a "sentence" and encode to vector in dimension 300
            Step : 
            1.) Word tokenize from "sentence"
            2.) C
    model =  SentenceTransformer('checkpoints/simcse-model-thai-version-supAIkeyword')reate a vector size == dimension
            3.) Add up the vector from the dictionary of index2word
            4.) return sentence vectorize
        """

        _w = sentence.split(' ')
        vec = np.zeros((1,dim))
        for word in _w:
            if (word in model.index_to_key):
                vec+= model.get_vector(word)
            else: pass
        if use_mean: vec /= len(_w)
        
        return vec

    def predict_class(self, feature_sentence, gram_sentence):

        # input_vec = self.count_vec.transform([sentence])
        tag = self.intent_model.predict(feature_sentence)
        return tag[0]


    def predict_score(self, feature_sentence, gram_sentence) -> list:
        """ predicted => list with array
        """

        predicted = self.intent_model.predict_proba(feature_sentence)
        pred = []
        ind = []
        
        for w in gram_sentence.split(' '):
            for idx, _intent in enumerate(kw):
                if w in kw[_intent] :
                    print("Word : {}, In config True !".format(w))
                    (predicted[idx])[0][1] = ((predicted[idx])[0][1] + self.weights_standout)/2
                    break
        for idx, val in enumerate(predicted):
            yes_score = val[0][1]

            if yes_score > self.confidence_score:
                # print("Yes score index number:{} {}".format(idx ,yes_score))
                ind.append(idx)
                pred.append(yes_score)
        return pred, ind

    def predict_tagging(self, clean_text : str, choice = 1):

        all_n_gram_phase = generate_n_gram(clean_text)
        print("Check all n gram phase : {}".format(all_n_gram_phase))

        tag_dict ={}

        for s in all_n_gram_phase :
            if choice == 1:
                s_vector = self.count_vec.transform([s])
            elif choice == 2:
                s_vector = self.word_embedded(self.wv_model, s)

            _score, _intent_idx = self.predict_score(s_vector, s)
            for ss, i_idx in zip(_score, _intent_idx):
                if (ss > self.confidence_score) : # Update to newer probability
                    if (self.tags[i_idx] not in list(tag_dict.keys())):
                        tag_dict.update({self.tags[i_idx] : _score}) # If there is keys on the dictionary

        print("DICT : {}" .format(tag_dict))        
        return tag_dict

