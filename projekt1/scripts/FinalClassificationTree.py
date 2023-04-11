from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from main_xy_encoded_for_part_c import *

reg_par_list =  (np.arange(0.5, 10, 1))
param_grid = {"max_depth":reg_par_list}
model = ensemble.RandomForestClassifier()
grid_search = GridSearchCV(model, param_grid, return_train_score = True, cv=10)
grid_search.fit(X,y)

res = grid_search.cv_results_

max_depth = grid_search.best_estimator_.__dict__["max_depth"]
max_depth

## The regression model
from sklearn.model_selection import KFold

# Function to crossval data
def crossval(fold, X, y):
    kf = KFold(n_splits=fold, shuffle=True,random_state = 10042023)
    for train, test in kf.split(X):
        trainX = X[train,:]
        testX = X[test,:]
        trainy = y[train]
        testy = y[test]
        yield testX, trainX, trainy, testy



# Run the outer cross val
for testX, trainX, trainy, testy in crossval(10,X,y):

        
    # Random forrest
    reg_par_list =  (np.arange(0.5, 10, 1))
    param_grid = {"max_depth":reg_par_list}
    model = ensemble.RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, return_train_score = True, cv=10)
    grid_search.fit(trainX, trainy)
    max_depth = grid_search.best_estimator_.__dict__["max_depth"] 
    
    ### Fit the final Linear regression with regularzation
    model = ensemble.RandomForestClassifier(max_depth = max_depth)
    fit = model.fit(trainX,trainy)
    pred = model.predict(testX)
    missclasified = len(np.where(pred != testy)[0])
    error = missclasified / len(testy)
    print(max_depth, "&",error,r"\\")
   



## Baseline
def baselinemodel(train, test):  
    most_seen_train = st.mode(train)[0][0]
    test_estimate = np.zeros(len(test))
    test_estimate[::] = most_seen_train
    missclasified = len(np.where(test_estimate != test)[0])
    E = missclasified / len(test)
    return E

### THe baseline model
for testX, trainX, trainy, testy in crossval(10,X,y):
    
    error_list =  []
    for testX, trainX, trainy, testy in crossval(10,trainX, trainy):
        error = baselinemodel(trainy, testy)
        error_list.append(error)
    print(min(error_list))






from scipy import stats as st

train = np.array([1,1,0,3,3,3])
test = np.array([1,3,3,3])

most_seen_train = st.mode(train)[0][0]

test_estimate = np.zeros(len(test))
test_estimate[::] = most_seen_train

missclasified = len(np.where(test_estimate != test)[0])
E = missclasified / len(test)
