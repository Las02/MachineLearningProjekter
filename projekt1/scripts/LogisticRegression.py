from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

from main_xy_encoded import *

# Train hyper parameters and plot
# We are using ridge regression which uses L2, just as in chp 14 in the book
reg_par_list =  (np.arange(0, 0.0008, 0.00005)) 
param_grid = {"C":reg_par_list}
model = LogisticRegression(multi_class="multinomial" ,solver='lbfgs')

K = 10
grid_results = []
for k in range(K):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=k)
    
    grid_search = GridSearchCV(model, param_grid,return_train_score = True, cv=10)
    grid_search.fit(X_train,y_train)
    res = grid_search.cv_results_
    grid_results.append(np.absolute(res["mean_test_score"]))
    
    

    sorted_df = pd.DataFrame({"error":np.absolute(res["mean_test_score"]),"lambda":reg_par_list}).sort_values(by="error")
    
    c = sorted_df["lambda"].values[0]
    model = LogisticRegression(multi_class="multinomial",solver='lbfgs',C = c)
    
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test,y_pred)
    grid_results.append(MSE)
    
    
    
    


plt.scatter(reg_par_list,np.absolute(res["mean_test_score"]))
plt.title("Generalization error for different parameters of Lambda")
plt.xlabel("Lambda")
plt.ylabel("Error")
#plt.xscale('log',base=10)