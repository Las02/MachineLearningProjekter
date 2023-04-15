from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

from main_xy_encoded_for_part_c import *



## The regression model
from sklearn.model_selection import KFold


# Train hyper parameters and plot
# We are using ridge regression which uses L2, just as in chp 14 in the book
reg_par_list =  (np.arange(0, 1, 0.1)) 
param_grid = {"C":reg_par_list}
model = LogisticRegression(multi_class="multinomial" ,solver='lbfgs',max_iter=10000)

       


K = 10
grid_results = []
contigency_table_regression = []
for k in range(K):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=k)
    
    grid_search = GridSearchCV(model, param_grid,return_train_score = True, cv=10)
    grid_search.fit(X_train,y_train)
    res = grid_search.cv_results_
    
    c = grid_search.best_estimator_.__dict__["C"]
    model = LogisticRegression(multi_class="multinomial",solver='lbfgs',C = c)
    
    model = model.fit(X_train,y_train)  
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test,y_pred)
    
    
    missclassified = len(np.where(y_pred != y_test)[0])
    error = missclassified / len(y_test)
    print(c, "&",error,r"\\")
    grid_results.append([c, error])
    
    for pr, test in zip(y_pred, y_test):
        contigency_table_regression.append(pr == test)

for c, error in grid_results:
    print(c, error)
