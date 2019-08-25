import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder,LabelEncoder 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from preprocessing_module import MultiLabelEncoder


mlt = MultiLabelEncoder()
loaded_model = joblib.load(open('Voting_Classifier.sav','rb'))


class input_pipeline():

    def __init__(self, dict_features):
        self.dict_features= dict_features

    def create_dataframe(dictf):
        df = pd.DataFrame(dictf, index=[1])
        return df

    def model(mlt_data):
        prediction = loaded_model.predict(mlt_data)
        return prediction
