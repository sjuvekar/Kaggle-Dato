import os
import cPickle

from feature_extractor import FeatureExtractor

class CachedFeatureExtractor(FeatureExtractor):
    
    def __init__(self, cache_path, batch_size=1000):
        self.cache_path = cache_path
        self.index = 0
        self.max_index = len(os.listdir(cache_path))

    def nextBatch(self):
        """
        Return a generator for a batch_size number of train (X, y) pairs from cache
        """
        for i in range(self.index, self.max_index):
            print "Reading batch ", i, "..."
            (X_train, y_train) = cPickle.load(open("{}/{}.pickle".format(self.cache_path, i)))
            yield (X_train, y_train)
        
