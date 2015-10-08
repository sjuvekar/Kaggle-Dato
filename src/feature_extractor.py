import cPickle
import numpy
import pandas

from sklearn.feature_extraction.text import HashingVectorizer

class FeatureExtractor(object):

    def __init__(self, csv_filename, batch_size = 1000):
        self.vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, non_negative=True)
        self.train_pd = pandas.read_csv(csv_filename)
        self.index = 0
        self.batch_size = batch_size

    def nextBatch(self):
        """
        Return a generator for a batch_size number of train (X, y) pairs
        """
        train_length = len(self.train_pd)
        while self.index < train_length:
            end_index = min(self.index + self.batch_size, train_length)
            print "Reading ", self.index, ": ", end_index
            X_train = list()
            y_train = list()

            for i in range(self.index, end_index):
                filename = "data/" + self.train_pd["file"][i]
                text = open(filename, "rb").readlines()
                X_train.append("\n".join(text))
                y_train.append(int(self.train_pd["sponsored"][i]))
            self.index = end_index
            yield (self.vectorizer.transform(X_train), y_train)

    def dumpBatch(self, batch, filename):
        with open(filename, "wb") as f:
            cPickle.dump(batch, f)

    def dump(self, filename):
        with open(filename, "wb") as f:
            cPickle.dump(self, f)

if __name__ == "__main__":
    # train
    f = FeatureExtractor("data/train.csv")
    count = 0
    for (X_train, y_train) in f.nextBatch():
        f.dumpBatch( (X_train, y_train), "working/models/train_batches/{}.pickle".format(count))
        count += 1
    f.dump("working/models/train_feature_extractor.pickle")

    # test
    f = FeatureExtractor("data/sampleSubmission.csv")
    count = 0
    for (X_train, y_train) in f.nextBatch():
        f.dumpBatch( (X_train, y_train), "working/models/test_batches/{}.pickle".format(count))
        count += 1
    f.dump("working/models/test_feature_extractor.pickle")
