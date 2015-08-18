import cPickle
import numpy
import pandas
from base_model import BaseModel

from operator import itemgetter
from sklearn.ensemble import GradientBoostingClassifier

class GBoostModel(BaseModel):

    def __init__(self, cached_features):
        BaseModel.__init__(self, cached_features)
        naive_bayes = cPickle.load(open("working/models/naive_bayes_model.pickle", "rb"))
        imp_neg_features = set([x[0] for x in sorted(enumerate(naive_bayes.model.feature_count_[0]), key=itemgetter(1))[-1000:]])
        imp_pos_features = set([x[0] for x in sorted(enumerate(naive_bayes.model.feature_count_[1]), key=itemgetter(1))[-1000:]])
        self.features = list(imp_neg_features.union(imp_pos_features))
        self.model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, subsample=0.7,
                                                min_samples_leaf=10, max_depth=7, random_state=1,
                                                verbose=3)
        self.X_train = pandas.DataFrame(columns = self.features)
        self.y_train = numpy.array([])

    def _fit_internal(self, X_train, y_train):
        X_new = pandas.DataFrame(X_train[:, self.features].toarray(), 
                                 columns = self.features)
        self.X_train = self.X_train.append(X_new)
        self.y_train = numpy.append(self.y_train, y_train)
        print "(X, y): (", self.X_train.shape, self. y_train.shape, ")"


    def fit(self):
        # First fit individual batches to create train matrix
        BaseModel.fit(self)
        # Finally, fit using model
        print "Calling GradientBoostingClassidier.fit "
        self.model.fit(self.X_train, self.y_train)


    def _predict_internal(self, X_test):
        X_new = pandas.DataFrame(X_test[:, self.features].toarray(), 
                                 columns = self.features)
        return self.model.predict_proba(X_new)[:, 1]


    def dump(self, filename):
        print "Dumping model to ", filename, "..."
        self.X_train = None
        self.y_train = None
        with open(filename, "wb") as f:
            cPickle.dump(self, f)
 
