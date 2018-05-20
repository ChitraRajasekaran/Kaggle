from pandas import read_csv
import pandas as pd
from sklearn import preprocessing
import numpy as np
from numpy import arange
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

train_dataset = read_csv("/Users/chitrasekar2k5/Desktop/Machine Learning/Kaggle submissions/train.csv",index_col = None, na_values = ['NA'])
test_dataset = read_csv("/Users/chitrasekar2k5/Desktop/Machine Learning/Kaggle submissions/test.csv",index_col = None,na_values = ['NA'])

test_pID = test_dataset['PassengerId']
"""target = train_dataset['Survived']"""
#analysis and preprocessing
train_dataset = train_dataset.drop(['Cabin'], axis = 1)
test_dataset = test_dataset.drop(['Cabin'], axis = 1)

median_age = train_dataset['Age'].median()
train_dataset['Age'] = train_dataset['Age'].fillna(median_age)
test_dataset['Age'] = test_dataset['Age'].fillna(median_age)

most_frequent_embarked = train_dataset['Embarked'].value_counts().index[0]
train_dataset['Embarked'] = train_dataset['Embarked'].fillna(most_frequent_embarked)
test_dataset['Embarked'] = test_dataset['Embarked'].fillna(most_frequent_embarked)

test_dataset['Fare'] = test_dataset['Fare'].fillna(train_dataset['Fare'].mean())

def preprocess_dataset(df):
    processed_dataset = df.copy()
    le = preprocessing.LabelEncoder()
    processed_dataset.Sex = le.fit_transform(processed_dataset.Sex)
    processed_dataset.Embarked = le.fit_transform(processed_dataset.Embarked)
    processed_dataset = processed_dataset.drop(['Name','Ticket','PassengerId'], axis = 1)
    return processed_dataset
processed_dataset = preprocess_dataset(train_dataset)

def preprocess_dataset1(df):
    processed_dataset = df.copy()
    le = preprocessing.LabelEncoder()
    processed_dataset.Sex = le.fit_transform(processed_dataset.Sex)
    processed_dataset.Embarked = le.fit_transform(processed_dataset.Embarked)
    processed_dataset = processed_dataset.drop(['Name','Ticket','PassengerId'], axis = 1)
    return processed_dataset
processed_dataset1 = preprocess_dataset1(test_dataset)

#split phase
X_train = processed_dataset.drop(['Survived'],axis = 1).values
Y_train = processed_dataset['Survived'].values
X_validation = processed_dataset1.values

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'
# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
#evaluate model in each turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',
LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Tune scaled KNN
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Tune scaled SVM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# ensembles
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C = 0.9, kernel = 'rbf')
model.fit(rescaledX, Y_train)
# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(predictions)
result_titanic = pd.DataFrame(data = {'PassengerId':test_pID, 'Survived': predictions})
result_titanic.to_csv('/Users/chitrasekar2k5/Desktop/Machine Learning/Kaggle submissions/titanic_results.csv',index = False)
