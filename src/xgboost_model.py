import os
import pandas
from base_model import BaseModel

class XGBoostModel(BaseModel):
    
    def __init__(self, cached_feature):
        BaseModel.__init__(self, cached_feature)
        
    def fit(self):
        os.system("$HOME/xgboost/xgboost xgboost.cfg")

    def predict_proba(self):
        os.system("$HOME/xgboost/xgboost xgboost.cfg task=pred model_in=working/models/test_5000.model")
        df = pandas.read_csv("pred.txt", header=None)
        return df[0]
