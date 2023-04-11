# exercise 8.2.5
from array import array
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats


import pandas as pd

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



# Set one hot encoding
y_len = len(y)
one_hot = np.zeros(shape=[y_len,3])
one_hot[:,0][y == 0] = 1
one_hot[:,1][y == 1] = 1
one_hot[:,2][y == 2] = 1



# Add attributes names
classNames = ["cytoplasm", "inner", "outer"]
attributeNames = ["mcg", "gvh", "aac", "alm1"] + classNames

#Extract y-vector again
tmp_y = X[:,4]
X = np.delete(X, 4, 1)
y = tmp_y


# Add class Names
# Get N .. ect TODO
N = len(y)
N = len(y)
M = len(attributeNames)


#Downsample: X = X[1:20,:] y = y[1:20,:]

X = stats.zscore(X,0)
    

X = np.concatenate((X, one_hot), axis=1)
GenErrors = dict()


# Parameters for neural network classifier
#n_hidden_units = 10    # number of hidden units
n_replicates = 1       # number of networks trained in each k-fold
max_iter = 10000         # stop criterion 2 (max epochs in training)

# K-fold crossvalidation
K = 10                  # only five folds to speed up this example

CV1 = model_selection.KFold(K, shuffle=True, random_state=100423)
CV2 = model_selection.KFold(5, shuffle=True, random_state=100423)
# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

BestErrorRate = 100
BestHiddenUnit = 0
BestHiddenUnitList = list()
BestErrorRateList = list()



for k1, (train_index, test_index) in enumerate(CV1.split(X,y)): 
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index]).unsqueeze(1)
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index]).unsqueeze(1)
    
    BestErrorRate = 100
    BestHiddenUnit = 0
    for n_hidden_units in [1,2,3,4,5]:
        errors = [] # make a list for storing generalizaition error in each loop
        for k, (train_index1, test_index1) in enumerate(CV2.split(X_train.squeeze(),y_train.squeeze())): 
            # Define the model, see also Exercise 8.2.2-script for more information.
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                                torch.nn.Sigmoid() # final tranfer function
                                )
            loss_fn = torch.nn.BCELoss()
    
            print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
            
            # Extract training and test set for current CV fold, convert to tensors
            X_train1 = torch.Tensor(X_train[train_index1,:])
            y_train1 = torch.Tensor(y_train[train_index1])
            X_test1 = torch.Tensor(X_train[test_index1,:])
            y_test1 = torch.Tensor(y_train[test_index1])
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train1,
                                                               y=y_train1,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            
            print('\n\tBest loss: {}\n'.format(final_loss))
            
            # Determine estimated class labels for test set
            y_sigmoid = net(X_test)
            y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8)
        
            # Determine errors and errors
            y_test = y_test.type(dtype=torch.uint8)
        
            e = y_test_est != y_test
            error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
            errors.append(error_rate) # store error rate for current CV fold 
        
            
        if np.mean(errors) < BestErrorRate:
            BestErrorRate = np.mean(errors)
            BestHiddenUnit = n_hidden_units
            BestTrainDataX = X_train1
            BestTrainDataY = y_train1
    
    
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, BestHiddenUnit), #M features to H hiden units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(BestHiddenUnit, 1), # H hidden units to 1 output neuron
                        torch.nn.Sigmoid() # final tranfer function
                        )
    loss_fn = torch.nn.BCELoss()
    
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=BestTrainDataX,
                                                       y=BestTrainDataY,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test)
    y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8)

    # Determine errors and errors
    y_test = y_test.type(dtype=torch.uint8)

    e = y_test_est != y_test
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    BestErrorRateList.append(round(float(error_rate),4)) # store error rate for current CV fold
    
    BestHiddenUnitList.append(BestHiddenUnit)
    

print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,3]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))
    




print(BestErrorRateList)
print(BestHiddenUnitList)

ANNError = [0.3529,0.3529,0.3529,0.3529,0.2941,0.2674,0.3939,0.5455,0.4242, 0.4242]
LinError = [0.3367, 0.174, 0.222, 0.1693, 0.4708, 0.3226,1.041, 0.5834, 0.753, 0.676]
BaseError = [0.6875,0.6941,0.6684, 0.6465, 0.5205, 0.5029, 0.5550, 0.5842, 0.6922, 0.6697]



stats.ttest_rel(ANNError,LinError)
stats.ttest_rel(ANNError,BaseError)
stats.ttest_rel(LinError,BaseError)

np.mean(LinError)
np.mean(BaseError)



