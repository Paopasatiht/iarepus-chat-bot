import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from utils.yamlparser import YamlParser

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
    print("----------Training process--------------")
    intent_model, prob_model = model_inititate(x_train, y_train)

    return intent_model, prob_model, tf_vectorizer

if __name__ == "__main__" :

    config_file = "/Projects/configs/config.yaml"
    cfg = YamlParser(config_file)
    data_corpus = pd.read_csv(cfg["DATA_CORPUS"]["data_csv"])
    
    intent_model, prob_model, tf_vectorizer= model_training(data_corpus) 

    # Testing process
    query_text = "รับคนเยอะมั้ย"
    x_input = tf_vectorizer.transform([query_text])
    print("Predicted class : {}".format(intent_model.predict(x_input)))

    # Predicted probability
    predicted = prob_model.predict_proba(x_input)
    max_ind = np.argmax(predicted,axis=1)
    print("Probability : {}".format(predicted[0][np.argmax(predicted)]))
