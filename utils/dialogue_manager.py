import numpy as np
import pandas as pd
from pythainlp import word_vector
from pythainlp import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


class DialogueManager():

    def __init__(self):
        """ dataset cols -> [Intents,Question, Answer,Question_vector]
        """
        
        self.model = word_vector.get_model()
        self.dataset = pd.read_csv("../iarepus-chat-bot/data_corpus.csv")
        
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

    def semantic_search(self, query_text):
        """ Return max_cos_ind and cosine_similarity value
        """

        query_vec = self.embedded(query_text)
        similar_lab = []

        #TODO : Fix here
        for i, q_vec in enumerate(self.QUESTION_VECTORS):
            q_vec = self.embedded(self.dataset.Question[i])
            sim = cosine_similarity(query_vec, q_vec)
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

if __name__ == "__main__":
    
    pass









    


