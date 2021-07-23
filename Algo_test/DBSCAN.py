import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN


# Importing the dataset
dataset = pd.read_csv('../Data_Set/Final/test.csv')
X = dataset.iloc[:, [2, 7]].values

clustering= DBSCAN(eps=10, min_samples=1,metric='euclidean').fit(X)

y_hc = clustering.fit_predict(X)
# plt.scatter(X[:,0], X[:,1],c=y, cmap='Paired')
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'green', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'red', label = 'Cluster 3')

# #filter rows of original data
# filtered_label0 = X[y_hc == 0]
 
# ##plotting the results
# plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])

# plt.scatter(X[:, 0], X[:, 1], c=y_hc, s=25, cmap='viridis')


plt.title('Density Based Spatial Clustering of Applications with Noise(DBCSAN)')
plt.xlabel('ALIGN')
plt.ylabel('Power')
plt.legend()
plt.show()

