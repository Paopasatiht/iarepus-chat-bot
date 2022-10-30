import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pythainlp import word_tokenize
from utils.yamlparser import YamlParser
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocess import generate_n_gram
from pythainlp.tag import pos_tag

config_file = "/Projects/configs/config.yaml"
cfg = YamlParser(config_file)
kw = cfg["KEYWORD_INTENT"]


class IntentsClassification():

    def __init__(self, word_vector_model, intent_model, count_vec, config_dict):

        self.intent_model = intent_model
        self.count_vec = count_vec
        self.wv_model = word_vector_model
        self.config_dict = config_dict
        self.tags = list(config_dict.keys())
        self.confidence_score = 0.40
        self.weights_standout = 0.70

    def word_embedded(self, _w : list, dim = 400, use_mean = True):
        """ Receive a "sentence" and encode to vector in dimension 300
        
        """
        
        # _w = word_tokenize(sentence)
        print("Reach here !")
        vec = np.zeros((1,dim))
        for word in _w:
            if (word in self.wv_model.index_to_key):
                vec+= self.wv_model.get_vector(word)
            else: pass
        if use_mean: vec /= len(_w)
        
        return vec

    def sentence_similarity(self, s1, s2):

        return cosine_similarity(self.word_embedded(str(s1)),self.word_embedded(str(s2)))

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

    def predict_tagging(self, clean_text : str, choice = 2):

        all_n_gram_phase = generate_n_gram(clean_text)
        print("Check all n gram phase : {}".format(all_n_gram_phase))

        tag_dict ={}

        for s in all_n_gram_phase :
            if choice == 1:
                s_vector = self.count_vec.transform([s])
            elif choice == 2:
                s_vector = self.word_embedded(s)
            _score, _intent_idx = self.predict_score(s_vector, s)
            for ss, i_idx in zip(_score, _intent_idx):
                if (ss > self.confidence_score) : # Update to newer probability
                    if (self.tags[i_idx] not in list(tag_dict.keys())):
                        tag_dict.update({self.tags[i_idx] : _score}) # If there is keys on the dictionary

        print("DICT : {}" .format(tag_dict))        
        return tag_dict

    def rule_base_tagging(self, sentence : str):
        """
        Input :
            text : str
                   clean text that passing preprocessing method
        Output :
            _    : dictionary
                   dictionary with tags as a "keys" and and score as "values" 
        
        """
        tag_dict = {}
        tokens = [token for token in word_tokenize(sentence, keep_whitespace=False) if token != ""]

        word_with_tag = pos_tag(tokens, corpus="orchid_ud")

        # Pull out only the word with "Verb tag"
        words = []
        for k in word_with_tag:
            if k[1] == "VERB":
                words.append(k[0])
        
        print("Verb word : {}".format(words))

        # Loop checking the "Most similarity" in "Configs dictionary"
        #TODO: fix here for reduct complexity
        for w in words:
            for item in self.config_dict.items():
                for x in item[1]:
                    sim = self.sentence_similarity(w, x)
                    print("Words in config : {}, prob : {}".format(x, sim))
                    if sim > self.confidence_score:
                        tag_dict.update({item[0] : sim})

    
        return tag_dict



