import cPickle
import numpy
import pandas
import sys

from feature_extractor import FeatureExtractor
from cached_feature_extractor import CachedFeatureExtractor

from sgd_model import SGDModel
from perceptron_model import PerceptronModel
from naive_bayes_model import NaiveBayesModel
from passive_aggressive_model import PassiveAggressiveModel
from gboost_model import GBoostModel
from keras_model import KerasModel
from lstm_model import LSTMModel
from convnet_model import ConvnetModel

if __name__ == "__main__":
    
    cached = True

    models = { "sgd_model": "SGDModel", 
               "perceptron_model": "PerceptronModel",
               "naive_bayes_model": "NaiveBayesModel",
               "passive_aggressive_model": "PassiveAggressiveModel",
               "gboost_model": "GBoostModel",
               "keras_model": "KerasModel",
               "convnet_model": "ConvnetModel"
    }

    model_name = sys.argv[1]
    print "Creating", model_name, ": ",
    model = globals()[models[model_name]](cached)
    print model.__class__.__name__

    model.fit()
    
    model.dump(sys.argv[2])


    
