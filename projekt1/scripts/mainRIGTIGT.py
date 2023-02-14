
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
for row in D.columns:
    if row not in ["lip","chg"]:
        D[row] = (D[row] - D[row].mean()) 

# Set binary values to 0 or 1
D["lip"][D["lip"] == 0.48]=0
D["lip"][D["lip"] == 1.00]=1
   
D["chg"][D["chg"] == 0.5]=0
D["lip"][D["lip"] == 1]=1     




# PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(D)
pca_df = pd.DataFrame(data=pca_results, columns=["PC1","PC2"])

# plot PCA
sns.scatterplot(x="PC1",y="PC2", data=pca_df, hue = tmp_y[8])
plt.legend(title="cell location", loc="upper left")
plt.show()
