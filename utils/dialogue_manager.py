import torch
import numpy as np
import pandas as pd

from pythainlp import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.pythainlp_utils import thai_bag_of_words, thai_tokenize

from utils.helper import _float_converter

class DialogueManager():

    def __init__(self,data_corpus, answer_model, intent_model,input_size, hidden_size, output_size, all_words, tags, device):
        """ dataset cols -> [Intents,Keys, Keys_vector,Values]
        """
        # Model && corpus initiate
        self.model = answer_model
        self.intent_tagging = intent_model
        self.dataset = data_corpus

        # Corpus parameter declarations
        self.QUESTION = self.dataset.Keys
        self.QUESTION_VECTORS = self.dataset.Keys_vector
        self.ANSWER = self.dataset.Values
        self.COSINE_THRESHOLD = 0.5
        self.CONF_SCORE = 0.65

        # Model parameters declaration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.all_words = all_words
        self.tags = tags
        self.device = device

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
        tag_dict ={}

        sentenced = thai_tokenize(sentenced)
        X = thai_bag_of_words(sentenced, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)

        output = self.intent_tagging(X)
        _, predicted = torch.max(output, dim=1)
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        tag_dict.update({self.tags[predicted.item()] : prob.item()})


        # Generate tagging where prob > CONF_SCORE
        # for idx, p in enumerate(probs[0]):
        #     if p > self.CONF_SCORE:
        #         tag_dict.update({self.tags[idx] : p.item()})

        return tag_dict

    def semantic_search(self, query_text):
        """ Search the matching question from the corpus, grasp the "keys" from the probability that passing criterion.
        "one intents" can answer only "one answer"
        # Step to generate a answer dictionary
        # 1.) Generate a key from tagging
        # 2.) Pick the "Key_vetor" from each "intent" and measure the similarity
        # 3.) If score > threshold, pick a values from "Values" columns and update dictionary as "intent" : "Values"
        # 4.) Redo with another intent
        """
        
        query_vec = self.sent_embeddings(query_text)
        
        most_relavance_dict= {}
        #Step 1 : Generate a key from tagging
        tag_dict = self.tagging(query_text)
        
        #Step 2 : Pick the key vector from each intent and measure the similarit 
        t = list(tag_dict)[0]
        print("Tagginh {}".format(t))
        answer_keys = self.dataset.loc[self.dataset.Intents == t].Keys_vector.tolist()
        for a_key in answer_keys:
            answer_vec = _float_converter(a_key)
            sim = cosine_similarity(query_vec, answer_vec)
            voting_prob = self.voting(tag_dict[t], sim)

            # 3.) If score > threshold, pick a values from "Values" columns and update dictionary as "intent" : "Values"
            if voting_prob > self.COSINE_THRESHOLD:
                most_relavance_dict.update({t : self.dataset.loc[self.dataset.Intents == t].Values.tolist()[0]})
                break
            else:
                pass


        # for t in tag_dict.keys():
        #     answer_keys = self.dataset.loc[self.dataset.Intents == t].Keys_vector.tolist() 
        #     for answer_key in answer_keys:
        #         answer_vec = _float_converter(answer_key)
        #         sim = cosine_similarity(query_vec, answer_vec)

        #         # Calculate a probability of two event
        #         voting_prob = self.voting(tag_dict[t], sim)

        #         #TODO : Add active learning loop at this point(optional)
        #         
        #         if voting_prob > self.COSINE_THRESHOLD:
        #             most_relavance_dict.update({t : self.dataset.loc[self.dataset.Intents == t].Values.tolist()[0]})
        #             sim_score.append(voting_prob)
        #             break
        #         else:
        #             pass


        return most_relavance_dict

    def voting(self, tag_prob : float, values_prob : float):
        """ Weighting the answer from "intent classification" and "Pattern matching"
        """

        return (tag_prob + values_prob) / 2


    def generate_answer(self, question):
        """ Query the matching "question" and return "answer"
        """
        answer_dict = self.semantic_search(question)

        if len(answer_dict) != 0:
            answer = ''
            for values in (answer_dict.values()): 
                                
                answer += "* " + values + "\n"
                
        else:
            answer = "น้อง Bot ไม่ค่อยเข้าใจความหมายเลยครับ ท่านสามารถตรวจสอบเพิ่มเติมได้ที่ https://superaiengineer2021.tawk.help/"
                                
            _f = open("logs/uncertainly_q.txt", "a")
            _f.write(question + "\n")
            _f.close()

        return answer

