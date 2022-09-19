import numpy as np
import pandas as pd
from pythainlp import word_vector
from pythainlp import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.helper import _float_converter

class DialogueManager():

    def __init__(self, model):
        """ dataset cols -> [Intents,Keys, Keys_vector,Values]
        """
        self.model = model
        self.dataset = pd.read_csv("../Projects/data_corpus_v2.csv")
        
        self.QUESTION = self.dataset.Keys
        self.QUESTION_VECTORS = self.dataset.Keys_vector
        self.ANSWER = self.dataset.Values
        self.COSINE_THRESHOLD = 0.5

    def embedded(self, sentence, dim = 300, use_mean = True):
        """ Receive a "sentence" and encode to vector in dimension 300
            Step : 
            1.) Word tokenize from "sentence"
            2.) Create a vector size == dimension
            3.) Add up the vector from the dictionary of index2word
            4.) return sentence vectorize
        """

        _w = word_tokenize(sentence)
        vec = np.zeros((1,dim))
        for word in _w:
            if word in self.model.index_to_key:
                vec+= self.model.get_vector(word)
            else: pass
        if use_mean: vec /= len(_w)
        
        return vec

    def sent_embeddings(self, sentenced : list):
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

    def semantic_search(self, query_text, multiple_answer = True):
        """ Search the matching question from the corpus, grasp the "keys" from the probability that passing criterion.
        "one intents" can answer only "one answer"
        """
        
        query_vec = self.sent_embeddings(query_text)
        sim_score = []
        most_relavance_dict= {}
        
        #TODO : Find the proper searching algorithms

        # for answer_vec in self.QUESTION_VECTORS:
            
        #     answer_vec = _float_converter(answer_vec)
        #     sim = cosine_similarity(query_vec, answer_vec)
        #     similar_score.append(sim)
        
        # if multiple_answer:
        #     p_index = [similar_score.index(val) for val in similar_score if val > self.COSINE_THRESHOLD]
        #     most_relavance_score = [similar_score[p] for p in p_index]

        _intent_ls = list(set(self.dataset.Intents.tolist()))
        
        for idx in range(len(_intent_ls)):
            answer_keys = self.dataset.loc[self.dataset.Intents == _intent_ls[idx]].Keys_vector.tolist() # Generate a list of key vector
            for answer_key in answer_keys:
                answer_vec = _float_converter(answer_key)
                sim = cosine_similarity(query_vec, answer_vec)
                #TODO : Add active learning loop at this point(optional)
                if sim > self.COSINE_THRESHOLD:
                    most_relavance_dict.update({_intent_ls[idx] : self.dataset.loc[self.dataset.Intents == _intent_ls[idx]].Values.tolist()[0]})
                    sim_score.append(sim)
                    break
                else:
                    pass


        return most_relavance_dict, sim_score

    def generate_answer(self, question):
        """ Query the matching "question" and return "answer"
        """
        answer_dict, sim_score = self.semantic_search(question)

        if len(answer_dict) != 0:
            answer = ''
            for idx, values in enumerate(answer_dict.values()): 
                                
                print("Probability : {}".format(sim_score[idx]))
                answer += "* " + values + "\n"
                
        else:
            answer = "น้อง Bot ไม่ค่อยเข้าใจความหมายเลยครับ ท่านสามารถตรวจสอบเพิ่มเติมได้ที่ https://superaiengineer2021.tawk.help/"
                                
            _f = open("logs/uncertainly_q.txt", "a")
            _f.write(question + "\n")
            _f.close()

        return answer

if __name__ == "__main__" :

    # For debug :
    q_vec = "โครงการที่จัดเป็นแบบไหน"
    model =  SentenceTransformer('mrp/simcse-model-roberta-base-thai')
    msg_manager = DialogueManager(model)

    resp = msg_manager.generate_answer(q_vec)
    print(resp)






    


