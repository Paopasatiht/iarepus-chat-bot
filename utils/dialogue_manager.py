import numpy as np
import pandas as pd
from pythainlp import word_vector
from pythainlp import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.helper import _float_converter

class DialogueManager():

    def __init__(self, model):
        """ dataset cols -> [Intents,Question, Answer,Question_vector]
        """
        self.model = model
        self.dataset = pd.read_csv("../Projects/data_corpus.csv")
        print("Load model please wait . . .")
        
        self.QUESTION = self.dataset.Question
        self.QUESTION_VECTORS = self.dataset.Question_vector
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
        """ embedding the sentenced base on the pre trained weights
        Parameters
            Input : 
                sentenced : string
                          : string that received from the user
            Output :
                answer_vec : list
                        : list of vector with fix dimension = 768
        """

        return self.model.encode([sentenced])

    def semantic_search(self, query_text):
        """ Return max_cos_ind and cosine_similarity value
        """

        # query_vec = self.embedded(query_text)
        query_vec = self.sent_embeddings(query_text)
        similar_lab = []
        
        for answer_vec in self.QUESTION_VECTORS:

            answer_vec = _float_converter(answer_vec)
            # a_vec = self.embedded(answer_vec)
            sim = cosine_similarity(query_vec, answer_vec)
            similar_lab.append(sim)

        max_ind = similar_lab.index(max(similar_lab))
        most_relavance_score = similar_lab[max_ind]

        return max_ind, most_relavance_score

    def generate_answer(self, question):
        """ Query the matching "question" and return "answer"
        """
        ind, most_relavance_score = self.semantic_search(question)
        print("Probability : {}".format(most_relavance_score))

        if most_relavance_score > self.COSINE_THRESHOLD:
            answer = self.dataset.Answer[ind]
        else:
            answer = "น้อง Bot ไม่ค่อยเข้าใจความหมายเลยครับ ท่านสามารถตรวจสอบเพิ่มเติมได้ที่ https://superaiengineer2021.tawk.help/"
            
            # If bot does not know sentence keep in the text:
            _f = open("logs/uncertainly_q.txt", "a")
            _f.write(question + "\n")
            _f.close()

        return answer





    


