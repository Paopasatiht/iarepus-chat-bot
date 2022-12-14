import pandas as pd
from pythainlp import word_vector
from sentence_transformers import SentenceTransformer 
from tqdm import tqdm
from utils.yamlparser import YamlParser

# Global declaration
config_file = "/Projects/configs/config.yaml"
cfg = YamlParser(config_file)
model = SentenceTransformer(cfg["MODEL"]["answer_model"])

def sentence_embedded(sentence : str):
    """ embedding the sentenced base on the pre trained weights
        Parameters
            Input : 
                sentenced : string
                          : string that received from the user
            Output :
                answer_vec : list
                        : list of vector with fix dimension = 768
        """
    return model.encode([sentence])

def generate_q_vector(df : pd.DataFrame):
    """ Read passsing through the Question in data corpus and convert to sentence vector in column name "Question_vector"
    """
    pbar = tqdm(total=len(df))
    q_vector = []
    for q in df.Keys:
        res = sentence_embedded(q)
        q_vector.append(res)
        pbar.update(1)

    df["Keys_vector"] = q_vector
    pbar.close()
    save_csv(df)
    print("Successfully generate Question vector !")

def save_csv(df: pd.DataFrame):
    """ Save new csv 
    """
    df.to_csv("/Projects/configs/data_corpus_v2.csv", float_format='%g')

if __name__ == "__main__":

    dataframe = pd.read_csv("/Projects/configs/data_corpus_v2.csv")
    generate_q_vector(dataframe)
