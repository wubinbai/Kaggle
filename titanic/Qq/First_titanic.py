#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


# In[2]:


df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')


# In[3]:


df_test.keys()


# In[4]:


df_train.keys()


# In[5]:


df_train.isna().sum()


# In[6]:


feature = df_train[['Age','Fare']]
target  = df_test[['Age','Fare']]
data = df_train.append(df_test,ignore_index=True,sort=False)


# In[7]:


df_train.describe()


# In[8]:


sns.countplot('Survived', hue = 'Sex', data = df_train)
df_train[['Sex','Survived']].groupby('Sex').mean()


# In[9]:


Sex_map = {
    'male':1,
    'female':0
}


# In[10]:


data['Sex'] = data['Sex'].map(Sex_map).astype('int')
df_train = data[:len(df_train)]
df_test  = data[len(df_train):]


# In[11]:


df_train[df_train['Sex'].isna()]


# In[12]:


sns.countplot('Survived', hue = 'Pclass', data = df_train)
df_train[['Pclass','Survived']].groupby('Pclass').mean()


# In[13]:


data.keys()


# In[14]:


df_train.Fare.max()


# In[15]:


df_train[df_train.Fare == 0][['Pclass','Survived','Fare']]    #guess fare = 0 is Crew


# In[16]:


data['Crew'] = 0                                                


# In[17]:


data['Crew'] = np.where(data['Fare'] == 0 , 1 , 0)


# In[18]:


data['Name_class'] = data['Name'].apply(lambda x : x.split(',')[1].split('.')[0])


# In[19]:


data['Name_class'].unique()


# In[20]:


title = []
mean_age = []
median_age = []
sex = []


# In[21]:


for i in data['Name_class'].unique():
    title.append(i)
    mean_age.append(round(data[data['Name_class'] == i]['Age'].mean()))
    median_age.append(round(data[data['Name_class'] == i]['Age'].median()))
    sex.append(data[data['Name_class'] == i]['Sex'].mean())


# In[22]:


dict = {'title':title, 'mean_age' : mean_age, 'median_age' : median_age, 'sex' : sex}


# In[23]:


pd.DataFrame(data = dict)


# In[24]:


for i in data['Name_class'].unique():
    index = data[(data['Name_class'] == i) & (data['Age'].isna())]['Age'].index
    for k in index:
        data.loc[index,'Age'] = round(data[data['Name_class'] == i]['Age'].median())


# In[25]:


data.Age.describe()


# In[26]:


data.Age.max(),data.Age.min()


# In[27]:


data['Age_class'] = data['Age']


# In[28]:


data['Age_class'] = data['Age_class'].apply(lambda x : 0 if x < 17 else 1 )


# In[29]:


data[data.Fare.isna()]


# In[30]:


data.loc[data.Fare.isna(),'Fare'] = data.loc[(data.Embarked == 'S') & (data.Pclass == 3),'Fare'].median()


# In[31]:


data['Fare_5'] = pd.qcut(data.Fare, 5)
data['Fare_5'] = data['Fare_5'].astype('category').cat.codes


# In[32]:


data.describe(include='all')


# In[33]:


data['Ticket_info'] = data['Ticket'].apply(lambda x : x.split(' ')[0] if not x.isdigit() else 'X')


# In[34]:


data['Ticket_count'] = 0


# In[35]:


z = []
for i,k in data.groupby('Ticket')['PassengerId']:              #利用groupby，把所有分類的丟到z
    z.append(k.values)


# In[36]:


for i in z:
    for k in i:                                                    #把每個票根的群組數目算出來
        data.loc[data['PassengerId'] == k,'Ticket_count'] = len(i)    


# In[37]:


sns.barplot(x = 'Ticket_count', y = 'Survived',data = data)


# In[38]:


data.Ticket_count = data.Ticket_count.apply(lambda x : 2 if x > 8 else 1 if (x>4 & x<9) | x == 1 else 0 )


# In[39]:


sns.barplot(x = 'Ticket_count', y = 'Survived',data = data)


# In[40]:


data.Ticket_count.astype('int32')


# In[41]:


data['Family'] = data['SibSp'] + data['Parch'] +1


# In[42]:


sns.barplot(data = data, x = 'Family', y = 'Survived')


# In[43]:


data['Family'] = data['Family'].apply(lambda x : 2 if x > 7 else 1 if (x>4) | x == 1 else 0)


# In[44]:


sns.barplot(data = data, x = 'Family', y = 'Survived')


# In[45]:


data.keys()
parameter = {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0}


# In[46]:


sns.barplot(data = data[(data.Pclass == 1)], x ='Embarked', y = 'Fare')


# In[47]:


data['Embarked'] = data['Embarked'].fillna('Q')          #依照在各個船艙中，為女生並且費用中位數最接近80的填入


# In[48]:


data['Embarked'] = data['Embarked'] .astype('category').cat.codes


# In[49]:


data.keys()


# In[50]:


data.Cabin.describe()


# In[ ]:





# In[51]:


df_train = data[:len(df_train)]
df_test  = data[len(df_train):]
feature = df_train[['Sex','Pclass']]
target  = df_test[['Sex','Pclass']]
model = XGBClassifier()
model.fit(feature,df_train['Survived'])
model.score(feature,df_train['Survived'])   
y_pred = model.predict(target).astype('int')
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],                #只提交 Sex、Pclass   0.75598
        "Survived": y_pred
    })
submission.to_csv('../submission1.csv', index=False)


# In[52]:


df_train = data[:len(df_train)]
df_test  = data[len(df_train):]
feature = df_train[['Sex','Pclass','Fare_5']]
target  = df_test[['Sex','Pclass','Fare_5']]
model = XGBClassifier()
model.fit(feature,df_train['Survived'])
model.score(feature,df_train['Survived'])   
y_pred = model.predict(target).astype('int')
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],                #只提交 Sex、Pclass、Fare_5   
        "Survived": y_pred                                    # 0.79425
    })
submission.to_csv('../submission2.csv', index=False)


# In[53]:


df_train = data[:len(df_train)]
df_test  = data[len(df_train):]
feature = df_train[['Sex','Pclass','Fare_5','Ticket_count']]
target  = df_test[['Sex','Pclass','Fare_5','Ticket_count']]
model = XGBClassifier()
model.fit(feature,df_train['Survived'])
model.score(feature,df_train['Survived'])   
y_pred = model.predict(target).astype('int')
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],                #只提交 Sex、Pclass、Fare_5、Ticket_count
        "Survived": y_pred                                    # 0.79425
    })
submission.to_csv('../submission3.csv', index=False)


# In[54]:


df_train = data[:len(df_train)]
df_test  = data[len(df_train):]
feature = df_train[['Sex','Pclass','Fare_5','Ticket_count','Family']]
target  = df_test[['Sex','Pclass','Fare_5','Ticket_count','Family']]
model = XGBClassifier()
model.fit(feature,df_train['Survived'])
model.score(feature,df_train['Survived'])   
y_pred = model.predict(target).astype('int')
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],                #只提交 Sex、Pclass、Fare_5、Ticket_count、Family
        "Survived": y_pred                                    # 0.79425
    })
submission.to_csv('../submission4.csv', index=False)


# In[144]:


df_train = data[:len(df_train)]
df_test  = data[len(df_train):]
feature = df_train[['Sex','Pclass','Fare_5','Ticket_count','Family','Crew']]
target  = df_test[['Sex','Pclass','Fare_5','Ticket_count','Family','Crew']]
model = XGBClassifier()
model.fit(feature,df_train['Survived'])
model.score(feature,df_train['Survived'])   
y_pred = model.predict(target).astype('int')
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],                #只提交 Sex、Pclass、Fare_5、Ticket_count、Family、Crew
        "Survived": y_pred                                    # 0.78947
    })
submission.to_csv('../submission5.csv', index=False)


# In[56]:


df_train = data[:len(df_train)]
df_test  = data[len(df_train):]
feature = df_train[['Sex','Pclass','Fare_5','Family']]
target  = df_test[['Sex','Pclass','Fare_5','Family']]
model = XGBClassifier()
model.fit(feature,df_train['Survived'])
model.score(feature,df_train['Survived'])   
y_pred = model.predict(target).astype('int')
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],                #只提交 Sex、Pclass、Fare_5、Family
        "Survived": y_pred                                    # 0.77511
    })
submission.to_csv('../submission6.csv', index=False)


# In[57]:


df_train = data[:len(df_train)]
df_test  = data[len(df_train):]
feature = df_train[['Sex','Pclass','Fare_5','Ticket_count','Family','Age_class','Crew']]
target  = df_test[['Sex','Pclass','Fare_5','Ticket_count','Family','Age_class','Crew']]
model = XGBClassifier()
model.fit(feature,df_train['Survived'])
model.score(feature,df_train['Survived'])   
y_pred = model.predict(target).astype('int')
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],                #只提交 Sex、Pclass、Fare_5、Ticket_count、Family、Crew
        "Survived": y_pred                                    # 0.77990
    })
submission.to_csv('../submission7.csv', index=False)


# In[58]:


pipe=Pipeline([('select',SelectKBest(k='all')), 
               ('classify', XGBClassifier(random_state = 10))])

param_test = {'classify__n_estimators':list(range(20,50,2)), 
              'classify__max_depth':list(range(3,60,3))}
gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10)
gsearch.fit(feature,df_train['Survived'])
print(gsearch.best_params_, gsearch.best_score_)


# In[59]:


df_train = data[:len(df_train)]
df_test  = data[len(df_train):]
feature = df_train[['Sex','Pclass','Fare_5','Ticket_count','Family']]
target  = df_test[['Sex','Pclass','Fare_5','Ticket_count','Family']]
model = XGBClassifier(n_estimators = 40, max_depth = 3)
model.fit(feature,df_train['Survived'])
model.score(feature,df_train['Survived'])   
y_pred = model.predict(target).astype('int')
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],                #只提交 Sex、Pclass、Fare_5、Ticket_count、Family
        "Survived": y_pred                                    # 0.79425
    })
submission.to_csv('../submission9.csv', index=False)


# In[60]:


model.score(feature,df_train['Survived'])


# In[ ]:


k_cv = StratifiedKFold(n_splits= 5, random_state= 42)


# In[66]:


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,cv = k_cv)


# In[67]:


results = cross_val_score(xgb1, feature, df_train['Survived'], cv=k_cv)


# In[68]:


results.mean(),results.std()


# In[69]:


for i in range(5,11):
    k_cv = StratifiedKFold(n_splits= i, random_state= 42)
    result = cross_val_score(xgb1, feature,  df_train['Survived'], cv=k_cv)
    print(i,result.mean(),result.std())


# In[157]:


feature = df_train[['Sex','Pclass','Fare_5','Ticket_count','Family','Age_class','Crew']]
target  = df_test[['Sex','Pclass','Fare_5','Ticket_count','Family','Age_class','Crew']]
model = XGBClassifier()
model.fit(feature,df_train['Survived'])
model.score(feature,df_train['Survived'])  


# In[160]:


parameter_grid1 = {
 'max_depth':[3,5,7,9],
 'min_child_weight':[1,3,5]
}


# In[161]:


grid_search = GridSearchCV(xgb1, param_grid = parameter_grid1, cv = 5)
grid_search.fit(feature,df_train['Survived'])
print("Best score:", grid_search.best_score_)
print("Best param:", grid_search.best_params_)


# In[162]:


parameter_grid2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[2,3,4]
}


# In[163]:


grid_search = GridSearchCV(xgb1, param_grid = parameter_grid2, cv = 5)
grid_search.fit(feature,df_train['Survived'])
print("Best score:", grid_search.best_score_)
print("Best param:", grid_search.best_params_)


# In[164]:


parameter_grid3 = {
 'gamma':[i/10.0 for i in range(0,5)],
}


# In[165]:


grid_search = GridSearchCV( XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,cv = k_cv), param_grid = parameter_grid3, cv = 5)
grid_search.fit(feature,df_train['Survived'])
print("Best score:", grid_search.best_score_)
print("Best param:", grid_search.best_params_)


# In[166]:


parameter_grid4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}


# In[167]:


grid_search = GridSearchCV( XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,cv = k_cv), param_grid = parameter_grid4, cv = 5)
grid_search.fit(feature,df_train['Survived'])
print("Best score:", grid_search.best_score_)
print("Best param:", grid_search.best_params_)


# In[168]:


parameter_grid5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}


# In[169]:


grid_search = GridSearchCV( XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,cv = k_cv), param_grid = parameter_grid5, cv = 5)
grid_search.fit(feature,df_train['Survived'])
print("Best score:", grid_search.best_score_)
print("Best param:", grid_search.best_params_)


# In[170]:


parameter_grid6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}


# In[171]:


grid_search = GridSearchCV( XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.75,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,cv = k_cv), param_grid = parameter_grid6, cv = 5)
grid_search.fit(feature,df_train['Survived'])
print("Best score:", grid_search.best_score_)
print("Best param:", grid_search.best_params_)


# In[174]:


model = XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=3,                                               
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.75,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,reg_alpha=0.00005,
 seed=27)
model.fit(feature,df_train['Survived'])
model.score(feature,df_train['Survived'])   
y_pred = model.predict(target).astype('int')
submission = pd.DataFrame({                                   # 修改超參數
        "PassengerId": df_test["PassengerId"],                #只提交 Sex、Pclass、Fare_5、Ticket_count、Family、Crew
        "Survived": y_pred                                    # 0.78947
    })
submission.to_csv('../submission11.csv', index=False)


# In[175]:


model.score(feature,df_train['Survived']) 


# In[177]:


test = XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=3,                                               
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.75,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,reg_alpha=0.00005,
 seed=27,cv = 5)


# In[178]:


for i in range(5,11):
    k_cv = StratifiedKFold(n_splits= i, random_state= 42)
    result = cross_val_score(test, feature,  df_train['Survived'], cv=k_cv)
    print(i,result.mean(),result.std())


# In[ ]:




