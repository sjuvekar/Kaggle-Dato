import cPickle
import numpy

from feature_extractor import FeatureExtractor
from cached_feature_extractor import CachedFeatureExtractor

class BaseModel(object):
    
    def __init__(self, cached_features=True):
        self.model = None
        self.classes = numpy.array([0, 1])
        self.cached_features = cached_features
        if cached_features:
            self.feature_extractor = CachedFeatureExtractor("working/models/train_batches")
            self.test_feature_extractor = CachedFeatureExtractor("working/models/test_batches")
        else:
            self.feature_extractor = FeatureExtractor("data/train.csv")
            self.test_feature_extractor = FeatureExtractor("data/test.csv")

    def fit(self):
        print "   Fitting ", self.__class__.__name__, "..."
        counter = 0
        for (X_train, y_train) in self.feature_extractor.nextBatch():
            self._fit_internal(X_train, y_train)
            if not self.cached_features:
                dump_file = "working/models/train_batches/{}.pickle".format(counter)
                print "Dumping to {}".format(dump_file)
                self.feature_extractor.dumpBatch((X_train, y_train), dump_file)
                counter += 1


    def _fit_internal(self, X_train, y_train):
        self.model.partial_fit(X_train, y_train, classes=self.classes)


    def predict_proba(self):
        print "   Predicting ", self.__class__.__name__, "..."
        counter = 0
        y_ans = numpy.array([])

        for (X_test, y_test_ignored) in self.test_feature_extractor.nextBatch():
            y_ans = numpy.append(y_ans, self._predict_internal(X_test))
            if not self.cached_features:
                dump_file = "working/models/test_batches/{}.pickle".format(counter)
                print "Dumping to {}".format(dump_file)
                self.test_feature_extractor.dumpBatch((X_test, y_test_ignored), dump_file)
                counter += 1
        return y_ans
   

    def _predict_internal(self, X_test):
        pass


    def dump(self, filename):
        print "Dumping model to", filename, "..."
        with open(filename, "wb") as f:
            cPickle.dump(self, f)
    
