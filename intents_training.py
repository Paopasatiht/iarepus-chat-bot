import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC

from utils.yamlparser import YamlParser
from pythainlp.word_vector import WordVector
from pythainlp import word_tokenize

from gensim.models import KeyedVectors
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from sentence_transformers import SentenceTransformer 

from utils.helper import get_th_tokens


import pickle
import os

SENT_EMB_MODEL =  SentenceTransformer("/Projects/checkpoints/simcse-model-thai-version-supAIkeyword")
os.environ["TOKENIZERS_PARALLELISM"] = "True"
# Reference : https://www.sbert.net/index.html?fbclid=IwAR2rsZExykBNvT4aWT1LJfq6diGkIPg1S1ZJ3ghTTf5xQfKsoREVhi-FA6k

def prepare_tf_feature(dataframe : pd.DataFrame,vectors : list, split=1):
    """ Prepare TF-feature
    """
    # features = tf_vectorizer.get_feature_names()

    x_train_counts = vectors[:len(dataframe)*split]
    x_test_counts = vectors[len(dataframe)*split:]
    y_train = dataframe.Intents[:len(dataframe)*split]
    y_test = dataframe.Intents[len(dataframe)*split:]

    return x_train_counts, y_train, x_test_counts, y_test, 

def prepare_embedded_feature(dataframe : pd.DataFrame, dim_size:int = 768):
    """ Load LTW2V model and encode dataframe.Keys as vectors, Return as X_train (list), y_train (list)
    """

    model = KeyedVectors.load_word2vec_format('checkpoints/LTW2V_v0.1.bin', binary=True, unicode_errors='ignore')
    print("load_embedded model finish")
    x_counts = []
    for x in dataframe.Keys :
        # vec = word_embedded(model, x)
        vec = SENT_EMB_MODEL.encode([x])
        print(vec.shape)
        x_counts.append(vec)

    
    x_train_counts = np.array(x_counts[:len(dataframe)])
    y_train = dataframe.Intents[:len(dataframe)]
    x_train_counts = x_train_counts.reshape((len(dataframe), dim_size))

    print("Finish prepare feature !")

    return x_train_counts, y_train
    

def word_embedded(model, sentence, dim = 400, use_mean = True) -> np.array:
        """ Receive a "sentence" and encode to vector in dimension 300
            Step : 
            1.) Word tokenize from "sentence"
            2.) C
    model =  SentenceTransformer('checkpoints/simcse-model-thai-version-supAIkeyword')reate a vector size == dimension
            3.) Add up the vector from the dictionary of index2word
            4.) return sentence vectorize
        """

        _w = word_tokenize(sentence, keep_whitespace=False)
        vec = np.zeros((1,dim))
        for word in _w:
            if (word in model.index_to_key):
                vec+= model.get_vector(word)
            else: pass
        if use_mean: vec /= len(_w)
        
        return vec

def prepare_feature(dataframe, vectors, choice = 1) :
    """ Create a feature feeding to ML model by,
    1 = TF-Vectors
    2 = Word embedding
    3 = Word embedding using bert
    """
    mlb = MultiLabelBinarizer()
    dataframe["Intents"] = dataframe["Intents"].apply(lambda x : [x])
    tags = mlb.fit_transform(dataframe['Intents'])

    if choice == 1 :
        x_train, y_train, x_test, y_test = prepare_tf_feature(dataframe, vectors)
        x_train, _, y_train, _ = train_test_split(x_train, tags, train_size=0.8, stratify=tags, random_state=42)
        print("--------------------Prepare TF feature. . . --------------------")
    elif choice == 2:
        x_train, y_train = prepare_embedded_feature(dataframe)
        print("--------------------Prepare embedded feature. . .===============================")
       
        print("Check tags : {}".format(mlb.classes_))
        dataframe["training_feature"] = dataframe["Intents"].copy()
        for idx, val in enumerate(x_train):
            dataframe["training_feature"][idx] = val
        x_train, _, y_train, _ = train_test_split(dataframe["training_feature"], tags, train_size=0.8, stratify=tags, random_state=42)
        x_train = x_train.to_list()
    else: pass

    return x_train, y_train

def model_inititate(x_train, y_train):
    """ Initite the training methodology with using GridSearch algorithms with training parameters
    """

    # Base estimator with multiOutput classifier
    RS=42
    _estimator = MultiOutputClassifier(SVC(class_weight='balanced', max_iter=10000,
                                             kernel='linear',random_state=RS, probability=True), n_jobs = -1)

    # Define the Gridsearch Parameters :
    param_grid = {"estimator__C" : [0.1, 1, 10, 100],
            'estimator__gamma': [1,0.1,0.01,0.001],
            'estimator__kernel': ['rbf', 'poly', 'sigmoid']
            }
    
    grid = GridSearchCV(_estimator, param_grid, refit = True, verbose = 3)
    clf = grid.fit(x_train, y_train)
    save_model(clf, "/Projects/checkpoints/intent-model-thai/TF_multioutput_linear_regress.pkl")

    return clf

def save_model(model,filepath : str, ):
    """ Save model with pickle lib
    """

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def model_training(dataframe : pd.DataFrame):
    """ Main function here Extract the feature using
    - 1.) TERM-FREQUENCY features (TF)
    - 2.) Word Embedded for training
    - 3.) etc
    """

    tf_vectorizer = CountVectorizer(tokenizer=get_th_tokens, ngram_range = (1, 2))
    vectors = tf_vectorizer.fit_transform(dataframe.Keys)
    
    # Prepare feature using tf vectors, word embedded
    x_train, y_train = prepare_feature(dataframe, vectors)
    
    print("----------Training process--------------")

    intent_model = model_inititate(x_train, y_train)

    return intent_model

if __name__ == "__main__" :

    config_file = "/Projects/configs/config.yaml"
    cfg = YamlParser(config_file)
    data_corpus = pd.read_csv(cfg["DATA_CORPUS"]["data_csv"])
    
    intent_model = model_training(data_corpus) 

    print("----------Finish Training process--------------")

