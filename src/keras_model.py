import cPickle
import base_model
import numpy
from operator import itemgetter

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
        naive_bayes = cPickle.load(open("working/models/naive_bayes_model.pickle", "rb"))
        imp_neg_features = set([x[0] for x in sorted(enumerate(naive_bayes.model.feature_count_[0]), key=itemgetter(1))[-10000:]])
        imp_pos_features = set([x[0] for x in sorted(enumerate(naive_bayes.model.feature_count_[1]), key=itemgetter(1))[-10000:]])
        self.features = list(imp_neg_features.union(imp_pos_features))
        self.model = self.SetupModel(len(self.features))
        self.n_epochs = 10


    def SetupModel(self, input_size=10000):
        model = Sequential()

        # deep pyramidal MLP, narrowing with depth
        model.add(Dropout(0.1))
        model.add(Dense(input_size, 512))
        model.add(PReLU((512,)))

        model.add(Dropout(0.075))
        model.add(Dense(512, 256))
        model.add(PReLU((256,)))

        model.add(Dropout(0.05))
        model.add(Dense(256, 128))
        model.add(PReLU((128,)))

        model.add(Dropout(0.025))
        model.add(Dense(128, 64))
        model.add(PReLU((64,)))

        model.add(Dropout(0.01))
        model.add(Dense(64, 32))
        model.add(PReLU((32,)))

        model.add(Dense(32, 2))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        return model


    def _fit_internal(self, X_train, y_train):
        X_new = X_train[:, self.features].toarray()
        y_new = np_utils.to_categorical(y_train)
        (loss, acc) = self.model.train_on_batch(X_new, y_new, accuracy=True)
        print "Loss = ", loss, ", Accuracy = ", acc


    def fit(self):
        print "   Fitting ", self.__class__.__name__, "..."
        numpy.random.seed(1337)
        for epoch in range(self.n_epochs):
            print "Training epoch", epoch, "..."
            counter = 0
            for (X_train, y_train) in self.feature_extractor.nextBatch():
                print "   Training batch ", counter, "...",
                self._fit_internal(X_train, y_train)
                if not self.cached_features:
                    dump_file = "working/models/train_batches/{}.pickle".format(counter)
                    print "Dumping to {}".format(dump_file)
                    seld.feature_extractor.dumpBatch((X_train, y_train), dump_file)
                counter += 1
       

    def predict_proba(self):
        print "   Predicting ", self.__class__.__name__, "..."
        print "Creating model using SetupModel..."
        self.model = self.SetupModel(len(self.features))
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


    def _predict_internal(self, X_test):
        X_new = X_test[:, self.features].toarray()
        return self.model.predict(X_new, batch_size=256)[:, 1]


    def dump(self, filename):
        print "saving weights to: ", self.weights_file
        self.model.save_weights(self.weights_file, overwrite=True)
        self.model = None
        print "Dumping model to ", filename, "..."
        with open(filename, "wb") as f:
            cPickle.dump(self, f)

