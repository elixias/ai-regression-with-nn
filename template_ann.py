import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer

dataset = pd.read_csv("50_Startups.csv")
#dataset = dataset.dropna()

target = dataset.iloc[:,-1].values
features = dataset.iloc[:,:-1].copy()

numerical_features = features.dtypes == 'float'
categorical_features = ~numerical_features

preprocess = make_column_transformer(
    (make_pipeline(SimpleImputer(), StandardScaler()), numerical_features),
    (OneHotEncoder(), categorical_features)
)
processedFeatures = preprocess.fit_transform(features);

sc = StandardScaler();
target = sc.fit_transform(target.reshape((-1,1)))

X_train, X_test, y_train, y_test = train_test_split(processedFeatures, target, random_state=0)

#print(X_train)
#print(y_train)

#modelling
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import cross_val_score

def build_classifier(optimizer='adam',loss='mean_squared_error'):
    classifier = Sequential()
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', input_dim = processedFeatures.shape[1]))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units = 1))
    classifier.compile(optimizer = optimizer, loss = loss)
    return classifier

cvclassifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [10,25],
              'epochs': [100,250,500],
              'optimizer': ['adam','sgd'],
			  'loss': ['mean_absolute_error','mean_squared_error']
			  }
grid_search = GridSearchCV(estimator = cvclassifier,
                           param_grid = parameters,
						   scoring='neg_mean_squared_error'
                           cv = 5)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)

newclassifier = Sequential()
newclassifier.add(Dense(units = 10, kernel_initializer = 'uniform', input_dim = processedFeatures.shape[1]))
newclassifier.add(Dropout(rate = 0.2))
newclassifier.add(Dense(units = 10, kernel_initializer = 'uniform'))
newclassifier.add(Dropout(rate = 0.2))
newclassifier.add(Dense(units = 1))
newclassifier.compile(optimizer = best_parameters["optimizer"], loss = best_parameters["loss"])
newclassifier.fit(X_train,y_train,batch_size=best_parameters["batch_size"],epochs=best_parameters["epochs"])

predicted = newclassifier.predict(X_test)
print(predicted)
print(y_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predicted)
print(mse)

print(sc.inverse_transform(y_test))
print(sc.inverse_transform(predicted))
print(">>>>>>>>>>>>END<<<<<<<<<<<<<")
"""exit

#predict
model = build_classifier(optimizer=best_parameters["optimizer"],loss=best_parameters["loss"])
model.fit(X_train,y_train,epochs=best_parameters["epochs"],batch_size=best_parameters["batch_size"])

predicted_y = model.predict(X_test)
print("Predicted y")
print(predicted_y)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predicted_y)

print(mse)
print(y_test)
"""

#redo with best score
#classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
#mean = accuracies.mean()
#variance = accuracies.std()
