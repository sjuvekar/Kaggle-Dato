import os
import pandas
from base_model import BaseModel

class XGBoostModel(BaseModel):
    
    def __init__(self, cached_feature):
        BaseModel.__init__(self, cached_feature)
        
    def fit(self):
        os.system("/spare/local/sjuvekar/xgboost/xgboost src/xgboost.cfg")

    def predict_proba(self):
        os.system("/spare/local/sjuvekar/xgboost/xgboost src/xgboost.cfg task=pred model_in=working/models/test_5000.model")
        df = pandas.read_csv("pred.txt", header=None)
        return df[0]

    def predict_and_dump(self, submission_file):
        df = self.predict_proba()
        samples_file = pandas.read_csv("data/sampleSubmission.csv")
        final_submission = pandas.DataFrame(columns = samples_file.columns)
        final_submission.file = samples_file.file
        final_submission.sponsored = df
        final_submission.to_csv(submission_file, index=False)
