
class SentEmbModel():

    def __init__(self, emb_model):

        self.emb_model = emb_model
    
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

        return self.emb_model.encode([sentenced])

    