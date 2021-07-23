import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


# Importing the dataset
data= pd.read_csv('../Data_Set/Final/test.csv')
X = data.iloc[:, [2,7]].values

clustering= KMeans(n_clusters=3, random_state=0).fit(X)

y= clustering.predict(X)
labels=clustering.labels_
cluster_centers=clustering.cluster_centers_

labels_unique=np.unique(labels)
n_clusters_=len(labels_unique)
print(y)



import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=7)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()