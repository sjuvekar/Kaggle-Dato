import cPickle
import numpy

from feature_extractor import FeatureExtractor
from cached_feature_extractor import CachedFeatureExtractor

class BaseModel(object):

    """
    This is the Base class for all models
    There are two ways of training any of the derived models:
    1) Online: override the _fit_internal() method that trains on a batch
    2) Complete: override the fit() method that trains on entire training set
    """
    def __init__(self, cached_features=True):
        self.model = None
        self.classes = numpy.array([0, 1])
        self.cached_features = cached_features
        self.feature_extractor = CachedFeatureExtractor("working/models/train_batches")
        self.test_feature_extractor = CachedFeatureExtractor("working/models/test_batches")
        

    def fit(self):
        print "   Fitting ", self.__class__.__name__, "..."
        counter = 0
        for (X_train, y_train) in self.feature_extractor.nextBatch():
            self._fit_internal(X_train, y_train)
            
    def _fit_internal(self, X_train, y_train):
        self.model.partial_fit(X_train, y_train, classes=self.classes)


    def predict_proba(self):
        print "   Predicting ", self.__class__.__name__, "..."
        counter = 0
        y_ans = numpy.array([])

        for (X_test, y_test_ignored) in self.test_feature_extractor.nextBatch():
            y_ans = numpy.append(y_ans, self._predict_internal(X_test))
        return y_ans
   

    def _predict_internal(self, X_test):
        pass


    def dump(self, filename):
        print "Dumping model to", filename, "..."
        with open(filename, "wb") as f:
            cPickle.dump(self, f)
    
