MODEL_DIR := working/models
PRED_DIR := working/predictions
VALIDATE_DIR := working/validations

all: $(PRED_DIR)/naive_bayes_model.csv $(PRED_DIR)/keras_model.csv $(PRED_DIR)/xgboost_model.csv

.PHONY: all

$(MODEL_DIR)/feature_extractor.pickle: src/feature_extractor.py
	python $^

$(MODEL_DIR)/libsvm_transformer.pickle: src/libsvm_transformer.py
	python $^

MODELS := naive_bayes_model keras_model xgboost_model

# Method to iterate over all models and build one-by-one
define make-model-targets

$(MODEL_DIR)/$(MODEL).pickle: src/train.py 
	python $$^ $(MODEL) $$@

$(PRED_DIR)/$(MODEL).csv: src/predict.py $(MODEL_DIR)/$(MODEL).pickle
	python $$^ $$@

endef

# Call method to build models one-by-one
$(foreach MODEL,$(MODELS),$(eval $(call make-model-targets,$MODEL)))

# Clean. touch both .py files
clean: src/train.py src/predict.py
	touch src/train.py src/predict.py
	rm *~ src/*~ src/*.pyc
