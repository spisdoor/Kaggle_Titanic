import pandas as pd
import numpy as np

train = pd.read_csv('train1.csv', index_col=None, na_values=['NA'])
test = pd.read_csv('test1.csv', index_col=None, na_values=['NA'])

# X = train[['Pclass', 'Sex_0', 'Sex_1', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Name_0', 'Name_1', 'Name_2', 'Name_3', 'Name_4', 'Embarked_0', 'Embarked_1', 'Embarked_2', 'Cabin_0', 'Cabin_1', 'Cabin_2', 'Cabin_3', 'Cabin_4', 'Cabin_5', 'Cabin_6', 'Cabin_7', 'Cabin_8']]
# y = train['Survived']
# data_test = test[['Pclass', 'Sex_0', 'Sex_1', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Name_0', 'Name_1', 'Name_2', 'Name_3', 'Name_4', 'Embarked_0', 'Embarked_1', 'Embarked_2', 'Cabin_0', 'Cabin_1', 'Cabin_2', 'Cabin_3', 'Cabin_4', 'Cabin_5', 'Cabin_6', 'Cabin_7', 'Cabin_8']]

X = train[['Pclass', 'Sex_0', 'Sex_1', 'Age', 'Fare', 'FamilySize', 'Name_0', 'Name_1', 'Name_2', 'Name_3', 'Name_4', 'Embarked_0', 'Embarked_1', 'Embarked_2']]
y = train['Survived']
data_test = test[['Pclass', 'Sex_0', 'Sex_1', 'Age', 'Fare', 'FamilySize', 'Name_0', 'Name_1', 'Name_2', 'Name_3', 'Name_4', 'Embarked_0', 'Embarked_1', 'Embarked_2']]

#################### Model ####################
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print ('--------------------')

print ('LogisticRegression')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
print (lr.score(x_test, y_test))
print ('--------------------')

print ('SVC')
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
print (svc.score(x_test, y_test))
print ('--------------------')

print ('RandomForestClassifier')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0)
rf.fit(x_train, y_train)
print (rf.score(x_test, y_test))
# for name, importance in zip(X.columns, rf.feature_importances_):
#     print(name, importance)
print ('--------------------')

print ('GradientBoostingClassifier')
from sklearn.ensemble import GradientBoostingClassifier
gdbt = GradientBoostingClassifier(n_estimators=30, max_depth=5, random_state=0)
gdbt.fit(x_train, y_train)
print (gdbt.score(x_test, y_test))
print ('--------------------')

print ('DecisionTreeClassifier')
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_test, y_test)
print(decision_tree.score(x_test, y_test))
# for name, importance in zip(X.columns, decision_tree.feature_importances_):
#     print(name, importance)
print ('--------------------')

print ('neighbors')
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()
knn.fit(x_test, y_test)
print(knn.score(x_test, y_test))
print ('--------------------')

print ('xgb')
import xgboost as xgb
xgbo = xgb.XGBClassifier()
xgbo.fit(x_train, y_train)
print (xgbo.score(x_test, y_test))
print ('--------------------')

print ('VotingClassifier')
from sklearn.ensemble import VotingClassifier  
# voting = VotingClassifier(estimators=[('lr', lr), ('svc', svc), ('rf', rf), ('gdbt', gdbt), ('decision_tree', decision_tree), ('knn', knn), ('xgbo', xgbo)])
voting = VotingClassifier(estimators=[('rf', rf), ('decision_tree', decision_tree), ('knn', knn)], voting='hard', weights=[1, 3, 1])
voting.fit(x_train, y_train)
print (voting.score(x_test, y_test) )
print ('--------------------')
#################### Model ####################

index = []
for i in range(892, 1310):
    index.append(i)

model = decision_tree
model.fit(X, y)
predictions = model.predict(data_test)
result = pd.DataFrame({'PassengerId': index, 'Survived': predictions})
result.to_csv('result.csv', index=False)
