import numpy as np
## SCATTER PLOT
import matplotlib.pyplot as plt
import random
## TSNE imports
from sklearn.manifold import TSNE


## EXTRACT THE FEATURES AND LABEL FROM TEXT FILE
data_file = 'data/pca_demo.txt'
file = open(data_file, 'r')
data = file.readlines()
no_of_columns = len(data[0].strip().split("\t"))

disease_list=[]
for x in data:
    disease_list.append(x.strip().split('\t')[no_of_columns-1])
matrix = np.loadtxt(data_file, delimiter="\t", usecols = range(no_of_columns - 1))
file.close()


## EXTRACT UNIQUE DISEASES AND MAP THEM TO AN INDEX KEY

# VARIABLES:
# matrix = all features
# disease_list = all labels

# extract label
disease_list_unique = np.unique(disease_list)

# create a map with unique keys for each unique disease
disease_map = dict()
i=0;
for disease in disease_list_unique:
    disease_map[disease]=i
    i+=1
disease_list_num = [disease_map[disease] for disease in disease_list]


## COMPUTE COVARIANCE MATRIX, EIGEN VALUE AND EIGEN VECTOR
## DIMENSIONAL REDUCTION TO 2D WITH COMPUTED EIGEN VALUE AND EIGEN VECTOR
## FIND PRINCIPAL COMPONENTS
tsne = TSNE(n_components=2, n_iter=500)
pca = tsne.fit_transform(matrix)
    

## SCATTER PLOT
col = []
for i in range(len(disease_list_unique)):
    col.append(plt.cm.jet(float(i)/len(disease_list_unique)))
for i,dis in enumerate(disease_list_unique):
    x = []
    y = []
    for it,xi in enumerate(pca[:,0]):
        if disease_list_num[it] == i:
            x.append(xi)
    for it,yi in enumerate(pca[:,1]):
        if disease_list_num[it] == i:
            y.append(yi)
    plt.scatter(x, y, c=col[i], label=dis)
plt.title("TSNE "+data_file + " scatter plot")
plt.legend()
plt.show()