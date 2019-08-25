""" Importing all packages that wewill use in building the model/ system"""
import pandas as pd
import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')
print('done importing packages')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
import sklearn
print(sklearn.__version__)
print('sklearn done!')

""" this is how we import the data we use to train and test the model """
df = pd.read_csv('PROJECTWork.csv')
# print(df.head())

test = pd.read_csv('BolaDola_test.csv')
# print(test.head())

df1= df.copy()


"""
drop the target/ label column
"""

df1_data = df1.drop("Categories of Malaria", axis=1)
df1_label = df1['Categories of Malaria']
print(df1_label.value_counts())


from preprocessing_module import MultiLabelEncoder
mlt = MultiLabelEncoder()
data2 = mlt.fit_transform(df1_data)
print(data2.head())
# print(type(data2))

encoder = OneHotEncoder()
ohe = encoder.fit(df1_data)
ohet = encoder.transform(df1_data)
# print(ohet)

# Logistic Regression
model = LogisticRegression(multi_class= 'auto')
model.fit(data2, df1_label)
print('the score for logistic regression: %f' % model.score(data2, df1_label))
print('the cross_val_score of logistic regression: %f ' % cross_val_score(model, X=data2,
                                                          y = df1_label,scoring='accuracy', cv=10).mean())
md = LogisticRegression(multi_class= 'auto')
md.fit(ohet, df1_label)
# print(md.score(ohet, df1_label))

# Decision Tree
tree = DecisionTreeClassifier(max_depth = 500)
tree.fit(data2, df1_label)
print('the score of the decision tree is %f'% tree.score(data2, df1_label))
print('the cross_val_score of decision trees is: %f ' % cross_val_score(tree, X=data2,
                                                                        y=df1_label, scoring='accuracy', cv=10).mean())

# Random Forest
model2 = RandomForestClassifier()
model2.fit(data2, df1_label)
print('the score for random forest is %f' % model2.score(data2, df1_label))
print('the cross_val_score of random forest: %f' % cross_val_score(model2, X=data2,
                                                         y= df1_label, scoring='accuracy', cv=10).mean())

# GaussianNB
model3 = GaussianNB()
model3.fit(data2, df1_label)
print('the score for GaussianNB: %f' % model3.score(data2, df1_label))
print('the cross val score of GaussianNb: %f' % cross_val_score(model3, X=data2, y=df1_label,
                                                                scoring='accuracy', cv=10).mean())

# Voting Classifier
vc = VotingClassifier(estimators=[
    ('Logistic Regression', model), ('Random Forest', model2),('GaussianNB', model3), ('Decision Tree', tree)], voting= 'soft')
vc.fit(data2, df1_label)
print('the score of voting classifier: %f' % vc.score(data2, df1_label))
print('the cross_val score of vc: %f' % cross_val_score(vc, X= data2, y= df1_label, scoring= 'accuracy',
                                                        cv=10).mean())




""" test data exploration """
data_test= test.copy()
data_data= data_test.drop("Categories of Malaria", axis=1)
datatest_label= data_test['Categories of Malaria']

dt= mlt.transform(data_data)

print('the test score is %f' % model2.score(dt, datatest_label))
print('the cross_val_score of the test data: %f' % cross_val_score(model2, X=dt, y=datatest_label,
                                                                    scoring='accuracy', cv=10).mean())
test_predicted= model2.predict(dt)
print(test_predicted)

# metrics exploration
results = confusion_matrix(datatest_label,test_predicted)
print('this is the confusion matrix score:', results)
print(classification_report(datatest_label, test_predicted))





# save the model(voting classifier and Random forest)
# voting classifer first
voting_model = 'Voting_Classifier.sav'
joblib.dump(vc, open(voting_model, 'wb'))

# random forest
random_model= 'Random_forest_classifier.sav'
joblib.dump(model2, open(random_model, 'wb'))


#logistic regression
log_reg= 'Logistic_reg.sav'
joblib.dump(md, open(log_reg, 'wb'))



