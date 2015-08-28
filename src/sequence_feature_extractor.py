import cPickle
import pandas
import sys
from collections import defaultdict

from HTMLParser import HTMLParser
from feature_extractor import FeatureExtractor
from tag_feature_extractor import TagFeatureExtractor, TagHTMLParser

class TagTransformer(HTMLParser):
    
    def __init__(self, feature_extractor_path):
        HTMLParser.__init__(self)
        self.extractor = cPickle.load(open(feature_extractor_path))
        self.n_tags = len(self.extractor.tag_dict)
        self.n_attrs = len(self.extractor.attr_dict)
        self.tag_dict = defaultdict(int, zip(sorted(self.extractor.tag_dict, reverse=True, key=lambda a: self.extractor.tag_dict[a]), range(1, self.n_tags+1)))
        self.attr_dict = defaultdict(int, zip(sorted(self.extractor.attr_dict, reverse=True, key=lambda a: self.extractor.attr_dict[a]), range(self.n_tags+1, self.n_tags+self.n_attrs+1)))
        self.neg_value = self.n_tags+self.n_attrs+1
        self.pos_value = self.n_tags+self.n_attrs+2
        print "Max = ", self.pos_value
        self.__reset__()

    def __reset__(self):
        self.X_train = list()
        
    def handle_starttag(self, tag, attr):
        self.X_train.append(self.tag_dict[tag])
        for (a, v) in attr:
            self.X_train.append(self.attr_dict[a])
            search_lookup = self.extractor.value_dict[v]
            if search_lookup <= 1:
                self.X_train.append(self.neg_value)
            else:
                self.X_train.append(self.pos_value)


class SequenceFeatureExtractor(FeatureExtractor):
    
    def __init__(self, csv_filename, batch_size = 1000):
        self.train_pd = pandas.read_csv(csv_filename)
        self.tag_transformer = TagTransformer("working/models/tag_train_batches/feature_extractor.pickle")
        self.index = 0
        self.batch_size = batch_size
        
    def nextBatch(self):
        train_length = len(self.train_pd)
        while self.index < train_length:
            end_index = min(self.index + self.batch_size, train_length)
            print "Reading ", self.index, ": ", end_index
            y_train = list()
            X_train = list()
            for i in range(self.index, end_index):
                filename = "data/" + self.train_pd["file"][i]
                text = ''.join(open(filename, "rb").readlines())
                self.tag_transformer.feed(text)
                if len(self.tag_transformer.X_train) == 0:
                    print "Appending html", self.tag_transformer.tag_dict["html"]
                    self.tag_transformer.X_train = [ self.tag_transformer.tag_dict["html"] ]
                X_train.append(self.tag_transformer.X_train)
                y_train.append(int(self.train_pd["sponsored"][i]))
                self.tag_transformer.__reset__()
            self.index = end_index
            yield (X_train, y_train)


if __name__ == "__main__":
    f = SequenceFeatureExtractor(sys.argv[1])
    counter = 0
    for (X_train, y_train) in f.nextBatch():
        dump_file = "working/models/{}/{}.pickle".format(sys.argv[2], counter)
        print "Dumping to {}".format(dump_file)
        f.dumpBatch((X_train, y_train), dump_file)
        counter += 1
