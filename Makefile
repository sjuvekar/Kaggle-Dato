MODEL_DIR := working/models
PRED_DIR := working/predictions
VALIDATE_DIR := working/validations

all: $(PRED_DIR)/naive_bayes_model.csv $(PRED_DIR)/passive_aggressive_model.csv $(PRED_DIR)/perceptron_model.csv $(PRED_DIR)/sgd_model.csv

.PHONY: all

MODELS := naive_bayes_model passive_aggressive_model perceptron_model sgd_model gboost_model keras_model convnet_model

# Method to iterate over all models and build one-by-one
define make-model-targets

$(MODEL_DIR)/$(MODEL).pickle: src/train.py 
	python $$^ $(MODEL) $$@

$(PRED_DIR)/$(MODEL).csv: src/predict.py $(MODEL_DIR)/$(MODEL).pickle
	python $$^ $$@

endef

# Call method to build models one-by-one
$(foreach MODEL,$(MODELS),$(eval $(call make-model-targets,$MODEL)))

# Save Sequence features for LSTM
$(MODEL_DIR)/sequence_train_batches: src/sequence_feature_extractor.py data/train.csv
	python $^ sequence_train_batches

$(MODEL_DIR)/sequence_test_batches: src/sequence_feature_extractor.py data/sampleSubmission.csv
	python $^ sequence_test_batches

# Clean. touch both .py files
clean: src/train.py src/predict.py
	touch src/train.py src/predict.py
	rm *~ src/*~ src/*.pyc
