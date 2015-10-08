import cPickle
import base_model
import numpy
from operator import itemgetter

numpy.random.seed(1337)
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

from cached_feature_extractor import CachedFeatureExtractor 

class LSTMModel(base_model.BaseModel):

    def __init__(self, cached_features):
        base_model.BaseModel.__init__(self, cached_features)
        self.weights_file = "working/models/lstm_model.pickle.weights"
        self.n_epochs = 4
        self.padding_size = 256
        self.max_features = 43980
        self.batch_size = 32
        self.model = self.SetupModel()
        self.X_train = None
        self.y_train = numpy.array([])
        self.feature_extractor = CachedFeatureExtractor("working/models/sequence_train_batches")
        self.test_feature_extractor = CachedFeatureExtractor("working/models/sequence_test_batches")

    def SetupModel(self):
        model = Sequential()
        model.add(Embedding(self.max_features, 128))
        model.add(LSTM(128, 128))
        model.add(Dropout(0.5))
        model.add(Dense(128, 1))
        model.add(Activation('sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
        return model


    def _fit_internal(self, X_train, y_train):
        if self.X_train is None:
            self.X_train = sequence.pad_sequences(X_train, maxlen=self.padding_size)
        else:
            self.X_train = numpy.vstack((self.X_train, sequence.pad_sequences(X_train, maxlen=self.padding_size)))
        self.y_train = numpy.append(self.y_train, y_train)
        print self.X_train.shape, self.y_train.shape

    def fit(self):
        print "Fitting ", self.__class__.__name__, "..."
        counter = 0
        for (X_train, y_train) in self.feature_extractor.nextBatch():
            print "   Adding batch ", counter, "...",
            self._fit_internal(X_train, y_train)
            counter += 1
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, nb_epoch=self.n_epochs, show_accuracy=True)

    def predict_proba(self):
        print "   Predicting ", self.__class__.__name__, "..."
        print "Creating model using SetupModel..."
        self.model = self.SetupModel()
        print "Loading model weights from ", self.weights_file, "..."
        self.model.load_weights(self.weights_file)
        counter = 0
        y_ans = numpy.array([])
        #self.test_feature_extractor = CachedFeatureExtractor("working/models/sequence_test_batches")
        for (X_test, y_test_ignored) in self.test_feature_extractor.nextBatch():
            y_ans = numpy.append(y_ans, self._predict_internal(X_test))
            counter += 1
        return y_ans


    def _predict_internal(self, X_test):
        X_new = sequence.pad_sequences(X_test, maxlen=self.padding_size)
        return self.model.predict(X_new, batch_size=self.batch_size)[:, 0]


    def dump(self, filename):
        print "saving weights to: ", self.weights_file
        self.model.save_weights(self.weights_file, overwrite=True)
        self.model = None
        print "Dumping model to ", filename, "..."
        with open(filename, "wb") as f:
            cPickle.dump(self, f)

