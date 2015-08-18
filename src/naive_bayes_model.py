from base_model import BaseModel

from sklearn.naive_bayes import MultinomialNB

class NaiveBayesModel(BaseModel):
    
    def __init__(self, cached_feature):
        BaseModel.__init__(self, cached_feature)
        self.model = MultinomialNB(alpha=0.01, fit_prior=True)

    def _predict_internal(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
