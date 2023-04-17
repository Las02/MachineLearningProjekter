from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy import stats as st
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.contingency_tables import mcnemar
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from main_xy_encoded_for_part_c import *


K = 10

reg_par_list =  (np.arange(1, 11, 1))
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

contingency_table_tree = []
tree_results = []
# Run the outer cross val
#for testX, trainX, trainy, testy in crossval(10,X,y):
for k in range(K):
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, random_state=k)

        
    # Random forrest
    reg_par_list =  (np.arange(1, 11, 1))
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
    print(max_depth,error)
    tree_results.append([max_depth,error])
    for pr, test in zip(pred, testy):
        contingency_table_tree.append(pr == test)


contingency_table_base = []

## Baseline
def baselinemodel(train, test):  
    most_seen_train = st.mode(train)[0][0]
    test_estimate = np.zeros(len(test))
    test_estimate[::] = most_seen_train
    missclasified = len(np.where(test_estimate != test)[0])
    E = missclasified / len(test)
    
    for pr, t in zip(test_estimate, test):
        contingency_table_base.append(pr == t)
    return E

### THe baseline model
#for testX, trainX, trainy, testy in crossval(10,X,y):
base_errors = []
for k in range(K):
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, random_state=k)
    
    
    #for testX, trainX, trainy, testy in crossval(10,trainX, trainy):
       
    error = baselinemodel(trainy, testy)
    base_errors.append((error))
        
    print((error))



# Train hyper parameters and plot
# We are using ridge regression which uses L2, just as in chp 14 in the book
reg_par_list =  (np.arange(0, 1, 0.1)) 
param_grid = {"C":reg_par_list}
model = LogisticRegression(multi_class="multinomial" ,solver='lbfgs',max_iter=10000)

       


K = 10
grid_results = []
contingency_table_regression = []
regression_results = []
for k in range(K):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=k)
    
    grid_search = GridSearchCV(model, param_grid,return_train_score = True, cv=10)
    grid_search.fit(X_train,y_train)
    res = grid_search.cv_results_
    
    c = grid_search.best_estimator_.__dict__["C"]
    model = LogisticRegression(multi_class="multinomial",solver='lbfgs',C = c,max_iter=10000)
    
    model = model.fit(X_train,y_train)  
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test,y_pred)
    
    
    missclassified = len(np.where(y_pred != y_test)[0])
    error = missclassified / len(y_test)
    print(c, "&",error,r"\\")
    regression_results.append([c,error])
    
    
    for pr, test in zip(y_pred, y_test):
        contingency_table_regression.append(pr == test)



    
    
def contingency_table(class1,class2):

    yy, yn, ny, nn = 0, 0, 0, 0
    
    for bol1,bol2 in zip(class1,class2):
        
        if bol1 == True and bol2 == True:
            yy += 1
        if bol1 == True and bol2 == False:
            yn += 1
        if bol1 == False and bol2 == True:
            ny += 1
        if bol1 == False and bol2 == False:
            nn += 1
    
    table = [[yy,yn],[ny,nn]]        
    return table


tree_regression = contingency_table(contingency_table_tree,contingency_table_regression)
tree_base =  contingency_table(contingency_table_tree,contingency_table_base)
regression_base =  contingency_table(contingency_table_regression, contingency_table_base)
                    
resulttr = mcnemar(tree_regression, exact=True)
resulttb = mcnemar(tree_base,exact = True)
resultrb = mcnemar(regression_base,exact = True)



print("tr = ",resulttr.pvalue)
print("tb = ", resulttb.pvalue)
print("rb = ", resultrb.pvalue)
    
print("Tree")
for l, error in tree_results:
    print(l, error)

print("Regression")
for c, error in regression_results:
    print(c, error)

print("Baseline")
for error in base_errors:
    print(error)

# Finding optimal c value
e_sum = 0
for c, error in regression_results:
    e_sum += c

lmbda = (e_sum/len(regression_results))
print(lmbda)
# Train hyper parameters and plot
# We are using ridge regression which uses L2, just as in chp 14 in the book
model = LogisticRegression(multi_class="multinomial",solver='lbfgs',C = lmbda ,max_iter=10000)

model = model.fit(X,y)  
#y_pred = model.predict(X_test)
model.coef_

# define the labels for the x-axis (features)
labels = attributeNames

# define the labels for the y-axis (classes)
classes = classNames


# create the heatmap
sns.heatmap(model.coef_, cmap='RdBu_r', annot=True, fmt='.2f', xticklabels=labels, yticklabels=classes)

# set the title
plt.title('Coefficients of Multinomial Logistic Regression')

# show the plot
plt.savefig('coefficients_plot.png', dpi=1000, bbox_inches='tight')

plt.show()


import scipy

def McNemar_conf_int(table):
    
    n11, n12, n21, n22 = table[0][0],table[0][1], table[1][0], table[1][1]
    
    n = (n11 + n12 + n21 + n22)
    
    E0 = (n12 - n12) / n
    Q = (n**2*(n+1)*(E0+1)*(1-E0))/(n*(n12+n21)-(n12-n21)**2)
    
    
    f = ((E0 + 1)/2) * (Q - 1)
    g = ((1 - E0)/2) * (Q - 1)
    
    alpha = 0.05
    
    theta_l = 2 * scipy.stats.beta.ppf(alpha/2, f, g) - 1
    theta_u = 2 * scipy.stats.beta.ppf(1-alpha/2,f,g) - 1
    
    return theta_l, theta_u

print("conf. int. tr", McNemar_conf_int(tree_regression))
print("conf. int. tb", McNemar_conf_int(tree_base))
print("conf. int. rb", McNemar_conf_int(regression_base))



#Accuracy of the random forest model (tree) and the regression
print("Tree")
tree_sum = 0
len(tree_results)
for l, error in tree_results:
    
    tree_sum += 1 - error

print(tree_sum / len(tree_results))

print("Regression")
reg_sum = 0
for c, error in regression_results:
    reg_sum += 1 - error

print(reg_sum/len(regression_results))




