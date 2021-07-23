from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd

data= pd.read_csv('../Data_Set/Final/test.csv')

x = pd.DataFrame(data.iloc[:,[2,7]].values).to_numpy()


kmeans = KMeans(n_clusters=3, random_state=0).fit(x)

y_kmeans = kmeans.predict(x)

# print(kmeans.cluster_centers_)
 
#filter rows of original data
filtered_label0 = x[y_kmeans == 0]
 
##plotting the results
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
plt.show()

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=25, cmap='viridis')

# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=20, alpha=0.5);


plt.xlabel('ALIGN')
plt.ylabel('Power')



