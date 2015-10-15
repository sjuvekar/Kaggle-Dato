This is the code for models that ended-up in top 10 in Kaggle's 'Dato - Truly Native?' contest

https://www.kaggle.com/c/dato-native

This code mainly builds XGBoost and MLP Keras models using HashingVectorizer (http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) features. Here are the steps to build the models:

* Assume that all data (i.e. all csv and html_txt files) is present in the data/ directory.
* Build HashingVectorizer features first using src/FeatureExtractor.py
* Learn a slightly-weak Naive-Bayes Classifier using all features
* Perform feature-extraction using Naive-Bayes feature importance
* Convert those important features in LibSVM format for easy sparse encoding
* Learn both XGBoost and Keras models using these features

