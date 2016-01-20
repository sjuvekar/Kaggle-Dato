# Extrct All features first, in batches of 1000
make working/models/feature_extractor.pickle

# Train a slightly weak Naives Bayes model on entire data
make working/models/naive_bayes_model.pickle

# Perform feature reduction and create sparse matrix using features selected
make working/models/libsvm_transformer.pickle

# Collect libsvm files in one file
cd working/models/svm
for fin in train_svm.txt.{0..337} ; do echo $fin; cat $fin >> train_svm.txt ; done
for fin in test_svm.txt.{0..66} ; do echo $fin; cat $fin >> test_svm.txt ; done
cd -

# Train the models now
make working/predictions/xgboost_model.csv
make working/predictions/keras_model.csv
