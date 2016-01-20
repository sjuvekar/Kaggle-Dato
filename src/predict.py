import cPickle
import numpy
import pandas
import sys

from naive_bayes_model import NaiveBayesModel
from keras_model import KerasModel
from xgboost_model import XGBoostModel
from libsvm_transformer import LibSvmTransformer

if __name__ == "__main__":
    
    model_file = sys.argv[1]
    print "Creating", model_file, ": ",
    model = cPickle.load(open(model_file))
    print model.__class__.__name__
    model.predict_and_dump(sys.argv[1])
