from base_model import BaseModel

from sklearn.linear_model import SGDClassifier

class SGDModel(BaseModel):
    
    def __init__(self, cached_features=True):
        BaseModel.__init__(self, cached_features)
        self.model = SGDClassifier(loss="modified_huber", average=True, random_state=1)

    def _predict_internal(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
