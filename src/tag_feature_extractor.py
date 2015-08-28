import re
import pandas
import sys

from HTMLParser import HTMLParser
from collections import defaultdict

from feature_extractor import FeatureExtractor

class TagHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.__reset__()

    def __reset__(self):
        self.tag_dict = defaultdict(int)
        self.attr_dict = defaultdict(int)
        self.value_dict = defaultdict(int)
        
    def handle_starttag(self, tag, attr):
        self.tag_dict[tag] += 1
        for (a, v) in attr:
            self.attr_dict[a] += 1
            self.value_dict[v] += 1

class TagFeatureExtractor(FeatureExtractor):

    def __init__(self, csv_filename, batch_size = 1000):
        self.train_pd = pandas.read_csv(csv_filename)
        self.index = 0
        self.batch_size = batch_size
        self.parser = TagHTMLParser()
        self.tag_dict = defaultdict(int)
        self.attr_dict = defaultdict(int)
        self.value_dict = defaultdict(int)
        self.collect_all_dicts = True

    def nextBatch(self):
        train_length = len(self.train_pd)
        while self.index < train_length:
            end_index = min(self.index + self.batch_size, train_length)
            print "Reading ", self.index, ": ", end_index
            y_train = list()

            for i in range(self.index, end_index):
                filename = "data/" + self.train_pd["file"][i]
                text = ''.join(open(filename, "rb").readlines())
                self.parser.feed(text)
                y_train.append(int(self.train_pd["sponsored"][i]))

            # Done, copy and reset parser
            if self.collect_all_dicts:
                for j in self.parser.tag_dict.keys():
                    self.tag_dict[j] += self.parser.tag_dict[j]
                for j in self.parser.attr_dict.keys():
                    self.attr_dict[j] += self.parser.attr_dict[j]
                for j in self.parser.value_dict.keys():
                    self.value_dict[j] += self.parser.value_dict[j]

            X_train = ( self.parser.tag_dict, self.parser.attr_dict, self.parser.value_dict )
            self.parser.__reset__()
            self.index = end_index
            yield (X_train, y_train)


if __name__ == "__main__":
    f = TagFeatureExtractor(sys.argv[1])
    counter = 0
    for (X_train, y_train) in f.nextBatch():
        dump_file = "working/models/{}/{}.pickle".format(sys.argv[2], counter)
        print "Dumping to {}".format(dump_file)
        f.dumpBatch((X_train, y_train), dump_file)
        counter += 1

    if f.collect_all_dicts:
        f.dump("working/models/tag_train_batches/feature_extractor.pickle")
