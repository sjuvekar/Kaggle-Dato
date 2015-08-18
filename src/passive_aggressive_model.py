from base_model import BaseModel

from sklearn.linear_model import PassiveAggressiveClassifier
from scipy.stats import logistic

class PassiveAggressiveModel(BaseModel):
    
    def __init__(self, cached_features):
        BaseModel.__init__(self, cached_features)
        self.model = PassiveAggressiveClassifier(loss='squared_hinge', C=1.0, random_state=1)

    def _predict_internal(self, X_test):
        return self.model.predict(X_test)
