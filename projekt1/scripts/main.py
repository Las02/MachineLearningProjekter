import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt




Data = pd.read_csv("../data/ecoli.data", sep = "\s+")

header = ["Seqname", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "cd"]
Data.columns = header

pca = PCA(n_components = 2)

Classifier = Data["cd"]

DataClean = Data.drop("Seqname", axis = 1)
DataClean = DataClean.drop("cd",axis = 1)

DataClean.loc[ DataClean["chg"] == 0.5, "chg"] = 0
DataClean.loc[ DataClean["lip"] == 0.48, "lip"] = 0


print(DataClean)
print(DataClean[["mcg","gvh","lip","chg","aac","alm1","alm2"]].describe().to_latex)
print(DataClean[["aac","alm1","alm2"]].describe().to_latex())

DataClean["cd"] = Data["cd"]
print(DataClean)
print(DataClean.groupby("cd")["chg"].describe().to_latex())
print(DataClean.groupby("cd")["lip"].describe().to_latex())
DataClean = DataClean.drop("chg",axis = 1)

tmp_y = list(Data["cd"])

sns.pairplot(DataClean, hue ="cd",diag_kind="hist", corner = True)
             
"""
fig, axs = plt.subplots(ncols=2,nrows = 3, figsize = (20,20))
sns.histplot(DataClean, x="mcg",ax=axs[0,0])
sns.histplot(DataClean, x="gvh",ax=axs[0,1])
sns.histplot(DataClean, x="lip",ax=axs[1,0])
sns.histplot(DataClean, x="aac",ax=axs[1,1])
sns.histplot(DataClean, x="alm1",ax=axs[2,0])
sns.histplot(DataClean, x="alm2",ax=axs[2,1])
"""


DataTotal = DataClean
DataTotal["cd"] = Classifier
print(DataTotal)
DataTotal.drop("lip", axis = 1)


print(DataTotal.groupby("cd").count())
DataTotal[DataTotal["cd"] == "imL"] = "im" 
DataTotal["cd"][DataTotal["cd"] == "imS"] = "im"
DataTotal["cd"][DataTotal["cd"] == "imU"] = "im"
DataTotal["cd"][DataTotal["cd"] == "pp"] = "om"
DataTotal["cd"][DataTotal["cd"] == "omL"] = "om"
DataTotal["cd"][DataTotal["cd"] == "imU"] = "om"
DataTotal["cd"][DataTotal["cd"] == "imL"]

