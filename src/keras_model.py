import sys
import pandas
import cPickle
import base_model
import numpy
import gc
import gzip
from operator import itemgetter
numpy.random.seed(1337)

from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.utils import np_utils

from cached_feature_extractor import CachedFeatureExtractor 

class KerasModel(base_model.BaseModel):

    def __init__(self, cached_features):
        base_model.BaseModel.__init__(self, cached_features)
        self.weights_file = "working/models/keras_model.pickle.weights"
        print "Loading naive_bayes.pickle..."
        naive_bayes = cPickle.load(open("working/models/naive_bayes_model.pickle", "rb"))
        print "Loading positive weights..."
        imp_neg_features = set([x[0] for x in sorted(enumerate(naive_bayes.model.feature_count_[0]), key=itemgetter(1))[-300000:]])
        print "Loading negative weights..."
        imp_pos_features = set([x[0] for x in sorted(enumerate(naive_bayes.model.feature_count_[1]), key=itemgetter(1))[-300000:]])
        self.features = list(imp_neg_features.union(imp_pos_features))
        del naive_bayes
        del imp_neg_features
        del imp_pos_features
        gc.collect()
        self.num_features = len(self.features)
        self.num_features = 408852
        self.model = self.SetupModel(self.num_features)
        self.n_epochs = 30


    def SetupModel(self, input_size=10000):
        model = Sequential()

        # deep pyramidal MLP, narrowing with depth
        model.add(Dropout(0.1, input_shape=(input_size,)))
        model.add(Dense(512, input_dim=input_size))
        model.add(PReLU())

        model.add(Dropout(0.1))
        model.add(Dense(256, input_dim=512))
        model.add(PReLU())

        model.add(Dropout(0.05))
        model.add(Dense(128, input_dim=256))
        model.add(PReLU())

        model.add(Dropout(0.025))
        model.add(Dense(64, input_dim=128))
        model.add(PReLU())

        model.add(Dense(2, input_dim=64))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        return model


    def _fit_internal(self, X_train, y_train):
        minibatch_size = 100
        start_index = 0
        X_len = X_train.shape[0]
        while start_index < X_len:
            end_index = min(start_index + minibatch_size, X_len)
            X_new = X_train[start_index:end_index, self.features].toarray()
            y_new = np_utils.to_categorical(y_train[start_index:end_index], nb_classes=2)
            print "   Training minibatch", start_index, ": ", end_index,
            (loss, acc) = self.model.train_on_batch(X_new, y_new, accuracy=True)
            print "Loss = ", loss, ", Accuracy = ", acc
            start_index = end_index
    

    def _predict_internal(self, X_test):
        y_ans = numpy.array([])
        minibatch_size = 100
        start_index = 0
        X_len = X_test.shape[0]
        while start_index < X_len:
            end_index = min(start_index + minibatch_size, X_len)
            X_new = X_test[start_index:end_index, self.features].toarray()
            y_ans = numpy.append(y_ans, self.model.predict(X_new, batch_size=minibatch_size)[:, 1])
            start_index = end_index
        return y_ans

    
    def fit(self):
        #self.model.load_weights("working/submit_models/keras_model.pickle.weights")
        print "   Fitting ", self.__class__.__name__, "..."
        shuffled_features = range(self.feature_extractor.max_index)
        numpy.random.shuffle(shuffled_features)
        # Create validation batches
        valid_batches = dict()
        for i in range(11):
            valid_batches[shuffled_features[i]] = 1
        print "These batches are used for validation: ", valid_batches.keys() 

        max_auc_score = 0
        for epoch in range(self.n_epochs):
            # Define test and validation arrays
            y_pred_total = numpy.array([])
            y_valid_total = numpy.array([])

            print "Training epoch", epoch, "..."
            shuffled_features = range(self.feature_extractor.max_index)
            numpy.random.shuffle(shuffled_features)
            for my_idx in shuffled_features:
                if my_idx in valid_batches.keys():
                    continue
                (X_train, y_train) = cPickle.load(open("{}/{}.pickle".format(self.feature_extractor.cache_path, my_idx)))
                print "   Training batch ", my_idx, "...",
                print "Positive Percentage =", sum(y_train) * 1. / len(y_train)
                self._fit_internal(X_train, y_train)

            for my_idx in valid_batches.keys():
                (X_train, y_train) = cPickle.load(open("{}/{}.pickle".format(self.feature_extractor.cache_path, my_idx)))
                print "   Collecting cross validation for batch ", my_idx, "..."
                y_pred_total = numpy.append(y_pred_total, self._predict_internal(X_train))
                y_valid_total = numpy.append(y_valid_total, y_train)

            auc_score = roc_auc_score(y_valid_total, y_pred_total)
            print "AUC score = ", auc_score
            if auc_score > max_auc_score:
                print "AUC score exceeded, saving weights"
                max_auc_score = auc_score
                self.model.save_weights(self.weights_file, overwrite=True)
    

    def predict_proba(self):
        print "   Predicting ", self.__class__.__name__, "..."
        print "Creating model using SetupModel..."
        self.model = self.SetupModel(self.num_features)
        print "Loading model weights from ", self.weights_file, "..."
        self.model.load_weights(self.weights_file)
        
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


    def dump(self, filename):
        print "saving weights to: ", self.weights_file
        #self.model.save_weights(self.weights_file, overwrite=True)
        self.model = None
        print "Dumping model to ", filename, "..."
        with open(filename, "wb") as f:
            cPickle.dump(self, f)

