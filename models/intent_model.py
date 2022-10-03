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

    def __init__(self, word_vector_model, intent_model, prob_model, count_vec):

        self.intent_model = intent_model
        self.prob_model = prob_model
        self.count_vec = count_vec
        self.confidence_score = 0.50
        self.wv_model = word_vector_model

    def word_embedded(self, word_ls : list, dim = 300, use_mean = True):
        """ Receive a "sentence" and encode to vector in dimension 300
            Step : 
            1.) Word tokenize from "sentence"
            2.) C
    model =  SentenceTransformer('checkpoints/simcse-model-thai-version-supAIkeyword')reate a vector size == dimension
            3.) Add up the vector from the dictionary of index2word
            4.) return sentence vectorize
        """

        _w = word_ls
        vec = np.zeros((1,dim))
        for word in _w:
            if word in self.wv_model.index_to_key:
                vec+= self.wv_model.get_vector(word)
            else: pass
        if use_mean: vec /= len(_w)

        return vec

    def predict_class(self, feature_sentence, gram_sentence):

        # input_vec = self.count_vec.transform([sentence])
        tag = self.intent_model.predict(feature_sentence)
        return tag[0]



    def predict_score(self, feature_sentence, gram_sentence):

        predicted = self.prob_model.predict_proba(feature_sentence)
        
        for w in gram_sentence.split(' '):
            for idx, _intent in enumerate(kw):
                if w in kw[_intent] :
                    predicted[0][idx] = (predicted[0][idx] + 1)/2
            max_ind = np.argmax(predicted, axis=1)
            pred = predicted[0][max_ind]
        return pred

    def predict_tagging(self, clean_text : str):

        all_n_gram_phase = generate_n_gram(clean_text)
        print("Check all n gram phase : {}".format(all_n_gram_phase))
        o_score = 0

        tag_dict ={}

        for s in all_n_gram_phase :
            
            s_vector = self.count_vec.transform([s])
            _intent = self.predict_class(s_vector, s)
            _score = self.predict_score(s_vector, s)
            if (_score > self.confidence_score) and (_score > o_score) : # Update to newer probability
                o_score = _score
                if (_intent not in list(tag_dict.keys())):
                    tag_dict.update({_intent : _score}) # If there is keys on the dictionary

        
        return tag_dict

