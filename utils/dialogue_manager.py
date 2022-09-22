import torch
import numpy as np
import pandas as pd

from pythainlp import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp_utils import thai_bag_of_words

from utils.helper import _float_converter
from utils.nltk_utils import bag_of_words

class DialogueManager():

    def __init__(self, answer_model, intent_model,input_size, hidden_size, output_size, all_words, tags):
        """ dataset cols -> [Intents,Keys, Keys_vector,Values]
        """
        # Model && corpus initiate
        self.model = answer_model
        self.intent_tagging = intent_model
        self.dataset = pd.read_csv("../Projects/configs/data_corpus_v2.csv")
        # Corpus parameter declarations
        self.QUESTION = self.dataset.Keys
        self.QUESTION_VECTORS = self.dataset.Keys_vector
        self.ANSWER = self.dataset.Values
        self.COSINE_THRESHOLD = 0.5
        # Model parameters declaration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.all_words = all_words
        self.tags = tags

    def word_embedded(self, sentence, dim = 300, use_mean = True):
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

        sentence = self.word_embedded(sentenced[0])
        X = thai_bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy()

        output = self.intent_tagging(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        return tag

    def semantic_search(self, query_text):
        """ Search the matching question from the corpus, grasp the "keys" from the probability that passing criterion.
        "one intents" can answer only "one answer"
        """
        
        query_vec = self.sent_embeddings(query_text)
        
        sim_score = []
        most_relavance_dict= {}
        
        _intent_ls = list(set(self.dataset.Intents.tolist()))
        
        # Step to generate a answer dictionary
        # 1.) Generate a key from "intent" columns
        # 2.) Pick the "Key_vetor" from each "intent" and measure the similarity
        # 3.) If score > threshold, pick a values from "Values" columns and update dictionary as "intent" : "Values"
        # 4.) Redo with another intent
        for idx in range(len(_intent_ls)):
            answer_keys = self.dataset.loc[self.dataset.Intents == _intent_ls[idx]].Keys_vector.tolist() 
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
    q_vec = "ต้องเสียค่าใช้จ่ายในการสมัครมั้ยครับ"
    model =  SentenceTransformer('checkpoints/simcse-model-thai-version-supAIkeyword')
    msg_manager = DialogueManager(model)

    resp = msg_manager.generate_answer(q_vec)
    print(resp)






    


