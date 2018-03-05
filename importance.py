import matplotlib
matplotlib.use('Agg')
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import pandas as pd

dataset = pd.read_csv('train1.csv', index_col=None, na_values=['NA'])

X = dataset[['Pclass', 'Sex_0', 'Sex_1', 'Age', 'FamilySize', 'Name_0', 'Name_1', 'Name_2', 'Name_3', 'Name_4', 'Embarked_0', 'Embarked_1', 'Embarked_2']]
# X = dataset[['Pclass', 'Sex_0', 'Sex_1', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Name_0', 'Name_1', 'Name_2', 'Name_3', 'Name_4', 'Embarked_0', 'Embarked_1', 'Embarked_2', 'Cabin_0', 'Cabin_1', 'Cabin_2', 'Cabin_3', 'Cabin_4', 'Cabin_5', 'Cabin_6', 'Cabin_7', 'Cabin_8']]
y = dataset['Survived']

model = XGBClassifier()
model.fit(X, y)

plot_importance(model)

pyplot.tight_layout()
pyplot.savefig('feature_importance.png')
