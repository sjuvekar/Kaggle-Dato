from base_model import BaseModel

from sklearn.linear_model import Perceptron
from scipy.stats import logistic

class PerceptronModel(BaseModel):
    
    def __init__(self, cached_features):
        BaseModel.__init__(self, cached_features)
        self.model = Perceptron(penalty="l2", random_state=1)

    def _predict_internal(self, X_test):
        return self.model.predict(X_test)
