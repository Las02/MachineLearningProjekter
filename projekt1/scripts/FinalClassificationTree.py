from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

from main_xy_encoded_for_part_c import *

reg_par_list =  (np.arange(0.5, 10, 1))
param_grid = {"max_depth":reg_par_list}


model = ensemble.RandomForestClassifier()
grid_search = GridSearchCV(model, param_grid, return_train_score = True, cv=10)
grid_search.fit(X,y)

res = grid_search.cv_results_
plt.scatter(reg_par_list,np.absolute(res["mean_test_score"]))
plt.title("Generalization error for different parameters of Lambda")
plt.xlabel("Lambda")
plt.ylabel("Error")


#Baseline
un = np.unique(y,return_counts = True)
print(un)