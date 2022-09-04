import pandas as pd
from pythainlp import word_vector
from utils.pythainlp_utils import sentence_vectorizer 


if __name__ == "__main__":

    df = pd.read_csv('data.csv')
    model = word_vector.get_model()
    _ls = []
    for val in df['Question']:
        print(val)
        _ls.append(sentence_vectorizer(val, model).astype(float))
        
    
    df['Question_vector'] = _ls

    print("Saving csv . . .")
    df.to_csv("/home/few-buntu/Projects/iarepus-chat-bot/data.csv")

    print(type(df['Question_vector'][0]))
