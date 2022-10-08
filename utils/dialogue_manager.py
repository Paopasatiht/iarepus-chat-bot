import torch
import numpy as np
import pandas as pd

from pythainlp import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.pythainlp_utils import thai_bag_of_words, thai_tokenize
from utils.database import DataStore
from utils.preprocess import preprocess_text
from models.intent_model import IntentsClassification
from utils.helper import _float_converter


class DialogueManager():

    def __init__(self,data_corpus, wv_model, answer_model, intent_model, prob_model, tf_vec, device, tags):
        """ dataset cols -> [Intents,Keys, Keys_vector,Values]
        """
        # Model && corpus initiate
        self.model = answer_model
        self.intent_tagging = IntentsClassification(wv_model,intent_model, prob_model, tf_vec, tags)
        self.dataset = data_corpus
        self.wv_model = wv_model

        # Corpus parameter declarations
        self.QUESTION = self.dataset.Keys
        self.QUESTION_VECTORS = self.dataset.Keys_vector
        self.ANSWER = self.dataset.Values
        self.COSINE_THRESHOLD = 0.35
        self.CONF_SCORE = 0.50

        self.device = device

        # Custom dictionary
        # self.custom_list = custom_ls

        # Database
        self.db = DataStore()

    def word_embedded(self, sentence, dim = 300, use_mean = True):
        """ Receive a "sentence" and encode to vector in dimension 300
            Step : 
            1.) Word tokenize from "sentence"
            2.) C
    model =  SentenceTransformer('checkpoints/simcse-model-thai-version-supAIkeyword')reate a vector size == dimension
            3.) Add up the vector from the dictionary of index2word
            4.) return sentence vectorize
        """

        _w = word_tokenize(sentence)
        vec = np.zeros((1,dim))
        for word in _w:
            if word in self.wv_model.index_to_key:
                vec+= self.wv_model.get_vector(word)
            else: pass
        if use_mean: vec /= len(_w)
        
        return vec

    def sent_embeddings(self, sentenced : str):
        """ embedding the sentenced base on the pre trained weights, bert embeddings
        Parameters
            Input : 
                sentenced : string
                          : string that received from the user
            Output :
                answer_vec : list
                        : list of vector with fix dimension = 768
        """

        return self.model.encode([sentenced])

    def tagging(self, sentenced : str):
        """ Receive a user input and predict the tags 
        Parameters
            Input :
                sentenced : string
                          : string that was receive from user
            Output :
                tag : string
                    : Refer to which class a query vector is, Reference from tag in configs/intents.json
        """
        tag_dict ={}

        sentenced = thai_tokenize(sentenced, self.custom_list)
        X = thai_bag_of_words(sentenced, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)

        output = self.intent_tagging(X)
        _, predicted = torch.max(output, dim=1)
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        print("Tagging : {}, prob {}".format(self.tags[predicted.item()], prob.item()))

        tag_dict.update({self.tags[predicted.item()] : prob.item()})

        return tag_dict

    def semantic_search(self, query_vec, clean_txt):
        """ Search the matching question from the corpus, grasp the "keys" from the probability that passing criterion.
        "one intents" can answer only "one answer"
        # Step to generate a answer dictionary
        # 1.) Generate a key from tagging
        # 2.) Pick the "Key_vetor" from each "intent" and measure the similarity
        # 3.) If score > threshold, pick a values from "Values" columns and update dictionary as "intent" : "Values"
        # 4.) Redo with another intent
        """
        
        most_relavance_dict= {}
        v_prob = []
        #Step 1 : Generate a key from tagging
        tag_dict = self.intent_tagging.predict_tagging(clean_txt) # -> Dictionary with all possible intent
        
        #Step 2 : Pick the key vector from each intent and measure the similarit 
        t = list(tag_dict.keys())
        pp = list(tag_dict.values())

        for t in tag_dict:
            answer_keys = self.dataset.loc[self.dataset.Intents == t].Keys_vector.tolist()
            
            for idx, a_key in enumerate(answer_keys):
                
                answer_vec = _float_converter(a_key)
                sim = cosine_similarity(query_vec, answer_vec)
                voting_prob = self.voting(tag_dict[t], sim)

                # 3.) If score > threshold, pick a values from "Values" columns and update dictionary as "intent" : "Values"
                if (voting_prob > self.CONF_SCORE) and (sim > self.COSINE_THRESHOLD):
                    most_relavance_dict.update({t : self.dataset.loc[self.dataset.Intents == t].Values.tolist()[0]}) # -> Dictionary with all possible intent
                    v_prob.append(voting_prob)
                    break
                else:
                    pass

        print("Tagging : {}, prob {}".format(most_relavance_dict.keys(), v_prob))

        return most_relavance_dict, v_prob

    def voting(self, tag_prob : float, values_prob : float):
        """ Weighting the answer from "intent classification" and "Pattern matching"
        """
        
        return (tag_prob[0] + values_prob[0][0]) / 2


    def generate_answer(self, question, debug=False):
        """ Query the matching "question" and return "answer"
        """
        clean_txt = preprocess_text(question)
        out_qavec = self.sent_embeddings(clean_txt)
        answer_dict, probability = self.semantic_search(out_qavec, clean_txt)
        
        
        if len(answer_dict) != 0:
            answer = ''
            for values in (answer_dict.values()): 
                                
                answer += "* " + values + "\n" + "                       " "\n"

            if ~debug:
                for idx, _i in enumerate(list(answer_dict.keys())):
                    self.db.push_to_database(_i, question, answer, probability[idx], str(out_qavec), status="pass")        
            
        else:
            answer = "ขอโทษนะค้าาา T^T น้อง Aeye ไม่ค่อยเข้าใจความหมายเลยค่ะ ท่านสามารถตรวจสอบเพิ่มเติมได้ที่ https://superai.aiat.or.th/ ได้เลยนะคะ"
            if ~debug:
                    self.db.push_to_database("unknown", question, answer, 0, str(out_qavec), status="fail")
          
            _f = open("logs/uncertainly_q.txt", "a")
            _f.write(question + "\n")
            _f.close()
        
        return answer
        
          

