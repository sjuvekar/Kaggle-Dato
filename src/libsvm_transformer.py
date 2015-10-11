import gc
import cPickle
import numpy
import pandas
from base_model import BaseModel

from operator import itemgetter
from sklearn.datasets import dump_svmlight_file

class LibSvmTransformer(BaseModel):

    def __init__(self, cached_features):
        BaseModel.__init__(self, cached_features)
        self.train_file = "working/models/svm/train_svm.txt"
        self.test_file = "working/models/svm/test_svm.txt"
        print "Reading naive_bayes..."
        naive_bayes = cPickle.load(open("working/models/naive_bayes_model.pickle", "rb"))
        print "Creating neg_features..."
        imp_neg_features = set([x[0] for x in sorted(enumerate(naive_bayes.model.feature_count_[0]), key=itemgetter(1))[-45000:]])
        print "Creating pos_features..."
        imp_pos_features = set([x[0] for x in sorted(enumerate(naive_bayes.model.feature_count_[1]), key=itemgetter(1))[-45000:]])
        del naive_bayes
        gc.collect()
        self.features = sorted(list(imp_neg_features.union(imp_pos_features)))
        print "features = ", self.features[0:20]
        del imp_neg_features
        del imp_pos_features
        gc.collect()
        self.count = 0
        self.test_count = 0

    def _convert_batch_to_libsvm(self, X, y, output_file):
        X_reduced = X[:, self.features]
        dump_svmlight_file(X_reduced, y, output_file)

    def _fit_internal(self, X_train, y_train):
        self._convert_batch_to_libsvm(X_train, y_train, "{}.{}".format(self.train_file, self.count))
        self.count += 1


    def _predict_internal(self, X_test):
        self._convert_batch_to_libsvm(X_test, numpy.zeros(len(X_test)), "{}.{}".format(self.test_file, self.test_count))
        self.test_count += 1


if __name__ == "__main__":
    l = LibSvmTransformer(True)
    
    # Transform train data
    print "Transforming Train data into libsvm format"
    for (X_train, y_train) in l.feature_extractor.nextBatch():
        i._fit_internal(X_train, y_train)
    
    # Transform test data
    print "Transforming Test data into libsvm format"
    for (X_test, y_test_ignored) in l.test_feature_extractor.nextBatch():
        l._predict_internal(X_test))
