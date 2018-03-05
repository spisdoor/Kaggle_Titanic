# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys

train = pd.read_csv('data/train.csv', na_values=['NA'])
test = pd.read_csv('data/test.csv', na_values=['NA'])
train = train.append(test)
train.reset_index(inplace=True, drop=True)

train['Survived'] = train['Survived'].fillna(0.0)
train.drop(['Ticket'], axis=1, inplace=True)

# Cabin
train['Cabin'] = train['Cabin'].fillna('N')
def replace_cabin(x):
  return str(x)[0]
train['Cabin'] = train.apply(replace_cabin)
train['Cabin'] = LabelEncoder().fit_transform(train['Cabin'])

# Name
def replace_name(x):  
  if ('Mrs' in x) or ('Lady' in x) or ('Mme' in x) or ('the Countess' in x):
    return 'Mrs'
  elif ('Mr' in x) or ('Rev' in x) or ('Sir' in x) or ('Major' in x) or ('Capt' in x) or ('Col' in x) or ('Don' in x) or ('Jonkheer' in x):
    return 'Mr'
  elif ('Miss' in x) or ('Mlle' in x) or ('Ms' in x):
    return 'Miss'
  elif ('Master' in x):
    return 'Master'
  elif ('Dr' in x):
    return 'Dr'
  else:
    print('Name Error')
train['Name'] = train['Name'].apply(replace_name)
train['Name'] = LabelEncoder().fit_transform(train['Name'])

# Sex
train['Sex'] = LabelEncoder().fit_transform(train['Sex'])

# Age
train['Age'] = train['Age'].fillna(-1)
# from sklearn.ensemble import RandomForestRegressor
# def predict_missing_age(data):
#   train1 = train[['Age', 'Pclass', 'Name', 'Parch', 'SibSp', 'Sex']]
#   train_age = train1[train.Age.notnull()].as_matrix()
#   test_age = train1[train.Age.isnull()].as_matrix()

#   X = train_age[:,1:]
#   y = train_age[:,0]

#   rf = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=50, max_features=2)
#   rf.fit(X, y)

#   # print(rf.score(X,y))
#   # print(rf.oob_score_)
#   result = rf.predict(test_age[:,1:])
#   # print(result)
#   data.loc[data.Age.isnull(), 'Age'] = result
#   return data
# train = predict_missing_age(train)

def replace_age(x):
  if (x >= 0) and (x <= 10):
    return 0
  elif (x >= 11) and (x <= 20):
    return 1
  elif (x >= 21) and (x <= 40):
    return 2
  else:
    return 3
train['Age'] = train['Age'].map(lambda x: replace_age(x))

# Fare
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())
def replace_fare(x):
  if (x < 60):
    return 0
  else:
    return 1
train['Fare'] = train['Fare'].map(lambda x: replace_fare(x))

# Embarked
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked'] = LabelEncoder().fit_transform(train['Embarked'])

# FamilySize
def get_family_size(x, y):
  if ((int)(x) + (int)(y) + 1 < 2):
    return 1
  elif (((int)(x) + (int)(y) + 1 >= 2) and ((int)(x) + (int)(y) + 1 <= 4)):
    return 2
  elif ((int)(x) + (int)(y) + 1 > 4):
    return 3
train['FamilySize'] = train.apply(lambda row: get_family_size(row['SibSp'], row['Parch']), axis=1)

train = pd.get_dummies(train, columns=['Name', 'Sex', 'Embarked', 'Cabin']).astype(np.int)

train.iloc[0:892,:].to_csv('train1.csv', index=False)
train.iloc[891:,:].to_csv('test1.csv', index=False)
