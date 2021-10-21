import pandas as pd
import numpy as np
import sklearn, re
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.model_selection import KFold, train_test_split 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv(r'Customer-Sales-Data.csv', thousands=',')

# This changes a column ('SIZE') of 'S', 'M', 'L' into 3 columns of the same name
# where the values are 0 or 1 so they can be used in the model (numbers only)
df = pd.get_dummies(df, columns=['SIZE'])

# If we have a few records that are missing price for some reason and we don't think it
# will derail our results, we can fill the NaN values with the column average
# Dataframes used for classification and regression models cannot have any
# NaN ("not a number") values
df['PRICE'].fillna(df['PRICE'].mean(), inplace=True)

# Removing any records that don't have a value for 'TAX'
df = df[~df['TAX'].isna()]

# Cutting dataframe to only records where the price is greater than or equal to $5
df = df[df['PRICE'] >= 5]

# Building a 70/30 train/test split from the dataframe
train, test = train_test_split(df, test_size=0.3)

# Building my test and train buckets from the split
# Target variable is a 0 or 1 in the column 'SALE' to indicate a purchase
X_train = train.drop('SALE', axis=1)
Y_train = train['SALE']
X_test = test.drop('SALE', axis=1)


# Build a Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

# Checking and visualizing my feature importances
importances = pd.DataFrame({'Feature':X_train.columns, 'Importance':np.round(random_forest.feature_importances_,2)})
importances = importances.sort_values(by='Importance', ascending=False)

# View results
importances

# Plot importances
importances['Importance'] = importances['Importance'].astype('float')
importances['Feature'] = importances['Feature'].astype('str')
fig=px.bar(importances, x='Feature', y='Importance', color='Feature')
fig.show()

# Cross validate results
from sklearn.model_selection import cross_val_score
scores = cross_val_score(random_forest, X_train, Y_train, cv=10, scoring='accuracy')
print ('Scores: ', scores)
print ('Mean: ', scores.mean())
print ('Standard Deviation: ', scores.std())

# Test some other models

# SGD
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# KNeighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Standard Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# Compare scores of all
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.reset_index(inplace=True)
result_df

# Calculate Biserial Correlations
biserial_dicts = []
for col in df.columns:
    biserial_dicts.append({'Feature':col, 
                           'Correlation':stats.pointbiserialr(df['SALE'], df[col])[0],
                           'P-value':stats.pointbiserialr(df['SALE'], df[col])[1]})
biserial_df = pd.DataFrame(data=biserial_dicts)

biserial_df.sort_values(by='Correlation', inplace=True)

biserial_df.reset_index(drop=True)

# Plot partial dependence
from sklearn.inspection import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.impute import SimpleImputer


df_y = df['SALE']
use_cols = ['SIZE', 'COLOR', 'PRICE', 'TAX', 'SHIPPING', 'INSTOCK']
df_X = df[use_cols]
# clf = GradientBoostingClassifier()
clf = RandomForestClassifier()
my_imputer = SimpleImputer()
imputed_df_X = my_imputer.fit_transform(df_X)
clf.fit(imputed_df_X, df_y)

my_plots = plot_partial_dependence(clf, features=[0,1,3], X=imputed_df_X, feature_names=use_cols, grid_resolution=8)