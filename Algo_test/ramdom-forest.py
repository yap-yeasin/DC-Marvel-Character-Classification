import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/home/yap/Documents/Code/ML/Final_Project/Data_Set/Final/test_predict.csv")
x = dataset.iloc[:,[2,3,4]].values
y = dataset.iloc[:, 5].values

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier  = RandomForestClassifier(n_estimators=10,random_state= 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
#y_pred2 = classifier.predict(x_train)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm= confusion_matrix(y_test, y_pred)
report= classification_report (y_test, y_pred)