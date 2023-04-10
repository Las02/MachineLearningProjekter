import pandas as pd
import numpy as np

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

y = [classDict[value[0]] for value in tmp_y.tolist()]

# Remove the "y" column from X
D = D.iloc[:,:7]

# Extract the ouput values to np matrix
np.array(D.values)
X = D.values

# shapes are correct
np.shape(X)
np.shape(y)

# Add attributes names
attributeNames = np.array(["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"])

# Add class Names
# Get N .. ect TODO
N = len(y)

N = len(y)
M = len(attributeNames)
classNames = np.array(["cytoplasm", "inner", "outer"])
C = len(classNames)
