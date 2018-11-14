# sum.py
####################
# sec 1
# What is ML?
####################
# sec 2
# Pros & Cons of Sklearn; 2 components of IPython Notebook: a) Kernel b) Browser; nbviewer.ipython.org is discussed
####################
# sec 3
# Equiv. saying:
# row: obs = sample = e.g. = instance = record
# column: feature = predictor = indep. var. = input = regressor = covariate
# val. we pred. = target = response = outcome = label = dep. var
####################
# sec 4
# knn: choose k, find k nearest, predict
# Paradigm: 4 steps: import, instantiate, train(fit), predict
####################
# sec 5
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
knn = KNeighborsClassifier(n_neighbors=1)
logreg = LogisticRegression()
# metrics
from sklearn.metrics import accuracy_score
# to use it:
# accuracy_score(y_true,y_pred)
# problem: e.g. train and test on the same data with knn has problem of overfitting, since it fits very well on known data, too complex model that cannot generalize to out-of-sample data.
# resolution: split data(train/test split)
# 3 Steps: a) split b)train on training se c) test on testing set
# import train/test split from sklearn
from sklearn.model_selection import train_test_split
# My Codes(read data)
from sklearn.datasets import load_iris
iris = load_iris()
X=iris.data
y=iris.target
# Use train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4)
# train(fit) with training set...
# test(accuracy_score) with testing set...
# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
want=input('Do you wannna print accuracy score for k=1 to 25 knn? Type "yes" or Enter to continue: ')
if want == 'yes':
    from pandas import Series
    s=Series(scores)
    print('accurarcy score is: ', s)
####################
# sec 6
import pandas as pd
from pandas import read_csv
data = read_csv('data/Advertising.csv', index_col = 0)# first column as index
data.head()
# data.tail()
data.shape
import seaborn as sns
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales')
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data['Sales']
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]
from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))
print(metrics.mean_squared_error(true, pred))
import numpy as np
print(np.sqrt(metrics.mean_squared_error(true, pred)))
####################
# sec 7
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=False).split(range(25))
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())
k_range = list(range(1, 31
))
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
# cross validation with logistic regression score
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())



	

