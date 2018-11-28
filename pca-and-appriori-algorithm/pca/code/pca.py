import numpy as np
## SCATTER PLOT imports
import matplotlib.pyplot as plt
import random


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

# compute mean of all rows
mean_vector = np.mean(matrix, axis=0)
# create a matrix with each column as the mean vector 
# means_expanded_vector = np.outer(mean_vector, np.ones(no_of_columns-1))
# Adjust original data with the mean
adjusted_orig_data = matrix - mean_vector
# compute covariance vector
covariance_matrix = np.cov(adjusted_orig_data.T)
# compute eigen values and eigen vectors
eigen_value, eigen_vector = np.linalg.eig(covariance_matrix)


## DIMENSIONAL REDUCTION TO 2D WITH COMPUTED EIGEN VALUE AND EIGEN VECTOR
## FIND PRINCIPAL COMPONENTS
dim_red =2
# sort in descending order eigen_value.argsort()[::-1]
# get the top dim_red values of the sorted array eigen_value.argsort()[:dim_red]
top_eigen_indeces = eigen_value.argsort()[::-1][:dim_red]
top_eigen_vectors = eigen_vector[:,top_eigen_indeces]
pca = np.empty([matrix.shape[0], top_eigen_vectors.shape[1]])
it = 0
for v in top_eigen_vectors.T:
    pca[:,it] = np.dot(adjusted_orig_data, v.T)
    it += 1


## SCATTER PLOT
col = []
for i in range(len(disease_list_unique)):
    col.append(plt.cm.jet(float(i)/len(disease_list_unique)))
# print('--- SCATTER PLOT ---')
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
plt.title(data_file + " scatter plot")
plt.legend()
plt.show()