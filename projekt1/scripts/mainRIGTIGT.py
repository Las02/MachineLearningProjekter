
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 
import seaborn as sns

D = pd.read_csv("../data/ecoli.data", header = None, sep="\s+")
# Remove ac numbers
D = D.iloc[:,1:]

# Set y
tmp_y=D[[8]]
# Remove the "y" column from X
D = D.iloc[:,:7]

# Set the attribute values
attributeNames = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]
D.columns = attributeNames

# Centering the data and standidizing 
# We are not doing it manually since its
# implemented in the method
for row in D.columns:
    if row not in ["lip","chg"]:
       D[row] = ((D[row]-D[row].mean()) / D[row].std()) 

# Set binary values to 0 or 1
D["lip"][D["lip"] == 0.48]=0
D["lip"][D["lip"] == 1.00]=1
   
D["chg"][D["chg"] == 0.5]=0
D["lip"][D["lip"] == 1]=1     

# Make subset without binary
D_nobin = D.drop(["lip","chg"],axis=1)

## Without binary
# PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(D_nobin)
pca_df = pd.DataFrame(data=pca_results, columns=["PC1","PC2"])

# plot PCA
sns.scatterplot(x="PC1",y="PC2", data=pca_df, hue = tmp_y[8])
plt.legend(title="cell location", loc="upper left")
plt.title("PCA of e-coli data colored for different cell locations")
plt.show()



## 3D pca
color_map = {'cp': 'red', 'im': 'blue', 'pp': 'green', 'imU': 'cyan', 'om': 'magenta', 'omL': 'yellow', 'imL': 'black', 'imS': 'orange'}
colors = [color_map[class_] for class_ in tmp_y[8]]

# Fit a PCA model to the data and transform the data to the new space
pca = PCA(n_components=5)
pca_data = pca.fit_transform(D_nobin)

# Create a 3D plot of the transformed data
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
print(tmp_y[8])

for class_ in color_map:
    indices = np.where(tmp_y[8] == class_)
    ax.scatter(pca_data[indices, 0], pca_data[indices, 1], pca_data[indices, 2], c=color_map[class_], label=class_)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend(loc='center left')
plt.title("3D PCA colored for different cell locations")
plt.show()

# Get culm variance
pca = PCA(n_components=5)
pca_results = pca.fit_transform(D_nobin)
cumsum_var = np.concatenate((np.array([0]),np.cumsum(pca.explained_variance_ratio_) *100))
plt.plot(["0","1","2","3","4","5"], cumsum_var)
plt.xlabel("Number of components")
plt.ylabel("Cumulative Explained variance (%)")
plt.ylim(0,100)
plt.xlim(0,5)
plt.axhline(y=90, color='r', linestyle='dotted')
plt.show()


## Getting the data for the vectors for the different pca's
pca.components_

eigenvectors = pca.components_.T
eigenvalues = pca.explained_variance_


corr_matrix = np.corrcoef(D_nobin, rowvar=False)

# Multiply the eigenvectors by the square root of the eigenvalues to obtain the loadings of the principal components
loadings = eigenvectors

# Plot the loadings as vectors in a scatter plot of the first two principal components
fig, ax = plt.subplots(figsize=(8,8))


for i, (x, y) in enumerate(zip(loadings[:,0], loadings[:,1])):
    ax.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.text(x*1.2, y*1.2, D_nobin.columns[i], ha='center', va='center', fontsize=12, color='g')
    #ax.text(x*1.4, y*1.4, "{:.2f}".format(corr_matrix[i,i]), ha='center', va='center', fontsize=10, color='r')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xlabel("PC1 ({}% of variance)".format(round(pca.explained_variance_ratio_[0]*100, 2)))
ax.set_ylabel("PC2 ({}% of variance)".format(round(pca.explained_variance_ratio_[1]*100, 2)))
plt.title("PCA Correlation Circle")

#circle = Circle((0,0), radius=1, fill=False, linestyle='--', color='gray', alpha=0.5)
#ax.add_artist(circle)

plt.show()