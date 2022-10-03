import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from utils.yamlparser import YamlParser
from pythainlp.word_vector import WordVector
from pythainlp import word_tokenize
wv = WordVector()

import pickle

def prepare_tf_feature(dataframe : pd.DataFrame,vectors : list, split=1):
    """ Prepare TF-feature
    """
    # features = tf_vectorizer.get_feature_names()

    x_train_counts = vectors[:len(dataframe)*split]
    x_test_counts = vectors[len(dataframe)*split:]
    y_train = dataframe.Intents[:len(dataframe)*split]
    y_test = dataframe.Intents[len(dataframe)*split:]

    return x_train_counts, y_train, x_test_counts, y_test, 

def prepare_embedded_feature(dataframe : pd.DataFrame):

    model = wv.get_model()
    # model = wv.get_model()
    x_counts = []
    for x in dataframe.Keys :
        vec = word_embedded(model, x)
        x_counts.append(vec)

    # x_new = x_counts[0]
    x_train_counts = np.array(x_counts[:len(dataframe)])
    # x_test_counts = np.array(x_counts[len(dataframe):])
    y_train = dataframe.Intents[:len(dataframe)]
    # y_test = dataframe.Intents[190:]

    x_train_counts = x_train_counts.reshape((len(dataframe), 300))
    # x_test_counts = x_test_counts.reshape((15, 300))

    return x_train_counts, y_train
    

def word_embedded(model, sentence, dim = 300, use_mean = True):
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
            if word in model.index_to_key:
                vec+= model.get_vector(word)
            else: pass
        if use_mean: vec /= len(_w)
        
        return vec

def model_inititate(x_train, y_train):

    # Model declaration
    text_classifier_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)

    # Define the Gridsearch Parameters :
    param_grid = {"alpha" : [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                'loss' : ['log_loss'],
                'penalty': ['l2'],
                'n_jobs': [-1]}
    
    grid = GridSearchCV(text_classifier_svm, param_grid, refit = True, verbose = 3)
    clf = grid.fit(x_train, y_train)
    save_model(clf, "/Projects/checkpoints/intent-model-thai/intent_model.pkl")

    # Calibrate and generate a probability model for SVM
    calibrator = CalibratedClassifierCV(clf, cv='prefit')
    model = calibrator.fit(x_train, y_train)
    save_model(model, "/Projects/checkpoints/intent-model-thai/prob_model.pkl")

    return clf, model

def save_model(model,filepath : str, ):

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def model_training(dataframe : pd.DataFrame):

    tf_vectorizer = CountVectorizer()
    vectors = tf_vectorizer.fit_transform(dataframe.Keys)
    
    x_train, y_train, x_test, y_test = prepare_tf_feature(dataframe, vectors)
    # x_train, y_train = prepare_embedded_feature(dataframe)
    print("----------Training process--------------")
    intent_model, prob_model = model_inititate(x_train, y_train)

    return intent_model, prob_model

if __name__ == "__main__" :

    config_file = "/Projects/configs/config.yaml"
    cfg = YamlParser(config_file)
    data_corpus = pd.read_csv(cfg["DATA_CORPUS"]["data_csv"])
    
    intent_model, prob_model= model_training(data_corpus) 
