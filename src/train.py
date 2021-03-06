import cPickle
import numpy
import pandas
import sys

from naive_bayes_model import NaiveBayesModel
from keras_model import KerasModel
from xgboost_model import XGBoostModel
from libsvm_transformer import LibSvmTransformer

if __name__ == "__main__":
    
    cached = True

    models = {"naive_bayes_model": "NaiveBayesModel",
              "keras_model": "KerasModel",
              "xgboost_model": "XGBoostModel",
              "libsvm_transformer": "LibSvmTransformer"
              }

    model_name = sys.argv[1]
    print "Creating", model_name, ": ",
    model = globals()[models[model_name]](cached)
    print model.__class__.__name__

    model.fit()
    
    model.dump(sys.argv[2])


    
