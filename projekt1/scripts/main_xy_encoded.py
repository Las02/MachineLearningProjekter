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
attributeNames = np.array(["mcg", "gvh", "aac", "alm1"] + classNames)

#Extract y-vector again
y = X[:,4]
X = np.delete(X, 4, 1)




# Add class Names
# Get N .. ect TODO
N = len(y)
N = len(y)
M = len(attributeNames)




