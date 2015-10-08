import cPickle
import keras_model
import numpy
from operator import itemgetter
numpy.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils

from cached_feature_extractor import CachedFeatureExtractor 

class ConvnetModel(keras_model.KerasModel):

    def __init__(self, cached_features):
        keras_model.KerasModel.__init__(self, cached_features)
        self.weights_file = "working/models/convnet_model.pickle.weights"
        self.num_features = len(self.features)
        self.model = self.SetupModel(self.num_features)
        self.n_epochs = 10

    def SetupModel(self, input_size=10000):
        model = Sequential()

        # ConvNet
        model.add(Convolution1D(input_size, 64, 3, border_mode='full')) 
        model.add(Activation('relu'))
        model.add(MaxPooling1D())
        model.add(Dropout(0.25))

        model.add(Convolution1D(32, 32, 3, border_mode='full')) 
        model.add(Activation('relu'))
        model.add(MaxPooling1D())
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(80, 20))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(20, 2))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        return model


    def _fit_internal(self, X_train, y_train):
        X_new = X_train[:, self.features].toarray().reshape(X_train.shape[0], 1, self.num_features)
        y_new = np_utils.to_categorical(y_train, nb_classes=2)
        (loss, acc) = self.model.train_on_batch(X_new, y_new, accuracy=True)
        print "Loss = ", loss, ", Accuracy = ", acc
        

    def _predict_internal(self, X_test):
        minibatch_size = 100
        X_new = X_test[:, self.features].toarray().reshape(X_test.shape[0], 1, self.num_features)
        y_ans = self.model.predict(X_new, batch_size=minibatch_size)[:, 1]
        return y_ans


