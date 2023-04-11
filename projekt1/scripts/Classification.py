


from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# requires data from main
from main_xy_encoded_for_part_c import *

# Fit logistic regression model

model = LogisticRegression(multi_class="multinomial",solver='lbfgs', C = 10)
model = model.fit(X,y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_probs = model.predict_proba(X)


# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))





# Print the accuracy of the model
accuracy = model.score(X, y)
print('Model accuracy:', accuracy)



