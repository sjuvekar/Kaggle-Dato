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

if __name__ == "__main__":
    
    cached = True

    models = { "sgd_model": SGDModel(cached), 
               "perceptron_model": PerceptronModel(cached),
               "naive_bayes_model": NaiveBayesModel(cached),
               "passive_aggressive_model": PassiveAggressiveModel(cached),
               "gboost_model": GBoostModel(cached),
               "keras_model": KerasModel(cached)
    }

    model_name = sys.argv[1]
    print "Creating", model_name, ": ",
    model = models[model_name]
    print model.__class__.__name__

    model.fit()
    
    model.dump(sys.argv[2])


    
