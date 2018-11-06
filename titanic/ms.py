import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=10)
logreg=LogisticRegression()

# Dealing with data
# Read train csv
tr_d=pd.read_csv('train.csv')
# Read test csv
te_d=pd.read_csv('test.csv')
# Get target and features of train data, y and X
y_tr=tr_d[tr_d.columns[1]]
X_tr=tr_d[tr_d.columns[:1].append(tr_d.columns[2:])]
# Test data has no target, and the feature is simply the data
X_te=te_d

# Features columns that I am choosing
feature_cols = ['PassengerId','Pclass','Age','SibSp','Parch','Fare']
# Re-assigning X and y using features I chose
X_tr=X_tr[feature_cols]
X_te=X_te[feature_cols]
y=y_tr

# conver NaN to Numbers
X_tr=np.nan_to_num(X_tr)
X_te=np.nan_to_num(X_te)
# Fit
knn.fit(X_tr, y)
logreg.fit(X_tr,y)
# Predict
pred_tr_knn=knn.predict(X_tr)
pred_tr_logreg=logreg.predict(X_tr)
pred_te_knn=knn.predict(X_te)
pred_te_logreg=logreg.predict(X_te)

# Find PassengerId:
l=list(X_te)
survived_knn=list(pred_te_knn)
survived_logreg=list(pred_te_logreg)
f=open('wubin_submission.csv','w+')
f.write('PassengerId,Survived\n')
for i in range(len(l)):
    PassId=str(int(l[i][0]))
    f.write(PassId)
    f.write(',')
    f.write(str(survived_knn[i]))
    f.write('\n')
f.close()

sc_tr=accuracy_score(y_tr,pred_tr_knn)
print('accuracy_score of training set, using knn, is: ', sc_tr)
sc_tr_logreg=accuracy_score(y_tr,pred_tr_logreg)
print('accuracy_score of training set, using logreg, is: ', sc_tr_logreg)
