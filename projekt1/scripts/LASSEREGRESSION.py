import pandas as pd
import numpy as np
from matplotlib.pylab import figure, plot, subplot, xlabel, ylabel, hist, show
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
D = pd.read_csv("../data/ecoli.data", header = None, sep="\s+")


# Remove ac numbers
D = D.iloc[:,1:]


# select y-vector 
tmp_y=D[[8]]
tmp_y = tmp_y.to_numpy()


# Extract y and class Names
classNames = sorted(np.unique(tmp_y))
classDict = dict(zip(classNames,range(len(classNames))))
classDict
classDict["imL"] = 1
classDict["imS"] = 1
classDict["imU"] = 1
classDict["im"] = 1
classDict["pp"] = 2
classDict["om"] = 2
classDict["omL"] = 2

y = np.array([classDict[value[0]] for value in tmp_y.tolist()])


# Remove the "y" column from X
D = D.iloc[:,:7]

# Extract the ouput values to np matrix
np.array(D.values)
X = D.values
X = np.delete(X,2,1)
X = np.delete(X,2,1)

# shapes are correct
np.shape(X)
np.shape(y)


X = X - np.mean(X, axis = 0)
X = X/np.std(X, axis = 0)

# Set one hot encoding
y_len = len(y)
one_hot = np.zeros(shape=[y_len,3])
one_hot[:,0][y == 0] = 1
one_hot[:,1][y == 1] = 1
one_hot[:,2][y == 2] = 1

X = np.concatenate((X, one_hot), axis = 1)

# Add attributes names
classNames = ["cytoplasm", "inner", "outer"]
attributeNames = np.array(["mcg", "gvh", "aac", "alm1"]+ classNames)

#Extract y-vector again
y = X[:,4]
X = np.delete(X, 4, 1)

# Add class Names
# Get N .. ect TODO
N = len(y)
N = len(y)
M = len(attributeNames)

# Train hyper parameters and plot
# We are using ridge regression which uses L2, just as in chp 14 in the book
reg_par_list =  10.**(np.arange(-2, 2, 0.1)) #np.arange(1,10**10,10**9)
param_grid = {"alpha":reg_par_list}
model = lm.Ridge()
grid_search = GridSearchCV(model, param_grid,return_train_score = True, cv=10)
grid_search.fit(X,y)
res = grid_search.cv_results_
plt.scatter(reg_par_list,-(res["mean_test_score"]))
plt.title("Generalization error for different parameters of Lambda")
plt.xlabel("Lambda")
plt.ylabel("Error")
plt.xscale('log',base=10) 
alpha = grid_search.best_estimator_.__dict__["alpha"]

grid_search
### Fit the final Linear regression with regularzation
model = lm.Ridge(alpha=alpha)
fit = model.fit(X,y)
pred = model.predict(X)

# Variables
intercept = fit.intercept_
coef = fit.coef_
values_coef = np.append(coef, intercept)
names_coef = np.append(attributeNames, "intercept")



for i in [4,5,6]:
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2) 
    ax1.scatter(x=X[:,0][X[:,i]==1], y=pred[X[:,i]==1])
    ax2.scatter(x=X[:,1][X[:,i]==1], y=pred[X[:,i]==1])
    ax3.scatter(x=X[:,2][X[:,i]==1], y=pred[X[:,i]==1])
    ax4.scatter(x=X[:,3][X[:,i]==1], y=pred[X[:,i]==1])
    ax1.set_ylabel("alm2")
    ax3.set_ylabel("alm2")
    ax3.set_title("\n     mcg")
    ax4.set_title("\n     gvh")
    ax3.set_xlabel("aac")
    ax4.set_xlabel("alm1")


### 1b
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


print(r"""
\begin{center}
\begin{tabular}{ |c|c| } 
\hline
lambda  &  error \\
""")
# Run the outer cross val
for testX, trainX, trainy, testy in crossval(10,X,y):

        
    # Run the inner cross val + training ect.
    reg_par_list =  10.**(np.arange(0, 4, 0.1)) #np.arange(1,10**10,10**9)
    param_grid = {"alpha":reg_par_list}
    model = lm.Ridge()
    grid_search = GridSearchCV(model, param_grid,return_train_score = True, cv=10)
    grid_search.fit(trainX,trainy)
    res = grid_search.cv_results_
    # Select the best alpha
    alpha = grid_search.best_estimator_.__dict__["alpha"]
    
    
    ### Fit the final Linear regression with regularzation
    model = lm.Ridge(alpha=alpha)
    fit = model.fit(trainX,trainy)
    pred = model.predict(testX)
    error = mean_squared_error(testy, pred, squared = False)
    print(alpha, "&",error,r"\\")
   
print(r""" \hline
\end{tabular}
\end{center}""")

def baselinemodel(train, test):  
    test_estimate = np.zeros(len(test))
    test_estimate[::] = np.mean(train)   
    test_estimate
    error = mean_squared_error(test_estimate, test, squared = False)
    return error

### THe baseline model
for testX, trainX, trainy, testy in crossval(10,X,y):
    
    error_list =  []
    for testX, trainX, trainy, testy in crossval(10,trainX, trainy):
        error = baselinemodel(trainy, testy)
        error_list.append(error)
    print("error", min(error_list))

