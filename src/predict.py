import cPickle
import pandas
import sys

if __name__ == "__main__":

   sample_file = pandas.read_csv("data/sampleSubmission.csv")
   model = cPickle.load(open(sys.argv[1]))

   y = pandas.DataFrame(columns = ["file", "sponsored"])
   y["file"] = sample_file["file"]
   y["sponsored"] = model.predict_proba()
    
   # Dump to submissions file
   print "Dumping results to", sys.argv[2], "..."
   y.to_csv(sys.argv[2], index=False)

   
