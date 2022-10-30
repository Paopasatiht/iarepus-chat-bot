from models.sent_emb_model import SentEmbModel
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

    def __init__(self,data_corpus, wv_model, sent_emb_model, intent_model, tf_vec, config_dict):
        """ dataset cols -> [Intents,Keys, Keys_vector,Values]
        """
        # Model && corpus initiate
        # self.model = answer_model
        self.sent_embedding = SentEmbModel(sent_emb_model)
        self.intent_tagging = IntentsClassification(wv_model,intent_model, tf_vec, config_dict)
        
        # Corpus declaration
        self.dataset = data_corpus
        self.tags = list(config_dict.keys())
        # Corpus parameter declarations
        self.QUESTION = self.dataset.Keys
        self.QUESTION_VECTORS = self.dataset.Keys_vector
        self.ANSWER = self.dataset.Values
        self.COSINE_THRESHOLD = 0.40
        self.CONF_SCORE = 0.60

        # Custom dictionary
        # self.custom_list = custom_ls

        # Database
        # self.db = DataStore()


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

        for t in tag_dict:
            answer_keys = self.dataset.loc[self.dataset.Intents == t].Keys_vector.tolist()
            
            for _, a_key in enumerate(answer_keys):
                
                answer_vec = _float_converter(a_key)
                sim = cosine_similarity(query_vec, answer_vec)
                voting_prob = self.voting(tag_dict[t], sim)

                # 3.) If score > threshold, pick a values from "Values" columns and update dictionary as "intent" : "Values"
                if (voting_prob > self.CONF_SCORE) and (sim > self.COSINE_THRESHOLD):
                    most_relavance_dict.update({t : self.dataset.loc[self.dataset.Intents == t].Values.tolist()[-1]}) # -> Dictionary with all possible intent
                    v_prob.append(voting_prob)
                    break
                else:
                    pass

        return most_relavance_dict, v_prob

    def voting(self, tag_prob : float, values_prob : float):
        """ Weighting the answer from "intent classification" and "Pattern matching"
        """
        
        return (tag_prob[0] + values_prob[0][0]) / 2


    def generate_answer(self, question, debug=False):
        """ Query the matching "question" and return "answer"
        """
        clean_txt = preprocess_text(question)
        out_qavec = self.sent_embedding.sent_embeddings(clean_txt)
        answer_dict, probability = self.semantic_search(out_qavec, clean_txt)
        
        
        if len(answer_dict) != 0:
            answer = ''
            for values in (answer_dict.values()): 
                                
                answer += "* " + values + "\n" + "                       " "\n"

            # if ~debug:
            #     for idx, _i in enumerate(list(answer_dict.keys())):
            #         self.db.push_to_database(_i, question, answer, probability[idx], str(out_qavec), status="pass")        
            
        else:
            answer = "ขอโทษนะค้าาา T^T น้อง Aeye ไม่ค่อยเข้าใจความหมายเลยค่ะ ท่านสามารถตรวจสอบเพิ่มเติมได้ที่ https://superai.aiat.or.th/ ได้เลยนะคะ"
            # if ~debug:
            #         self.db.push_to_database("unknown", question, answer, 0, str(out_qavec), status="fail")
          
            _f = open("logs/uncertainly_q.txt", "a")
            _f.write(question + "\n")
            _f.close()
        
        return answer
        
          

