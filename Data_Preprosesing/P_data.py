import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

## Importing the dataset & reading two csv files
# data1 = pd.read_csv('/home/yap/Documents/Code/ML/Final_Project/Data_Set/Final/Marvel_Data.csv',usecols = [2,4,5,6,7,8,10,14])
# data2 = pd.read_csv('/home/yap/Documents/Code/ML/Final_Project/Data_Set/Final/DC_Data.csv',usecols = [2,4,5,6,7,8,10,14])

# ### Combined the dataset
# data = pd.concat([data1, data2])

data = pd.read_csv("/home/yap/Documents/Code/ML/Final_Project/Data_Set/Final/test.csv")

## Label_Encoding

cols = ['HAIR','EYE','ALIVE']

data[cols] = data[cols].apply(LabelEncoder().fit_transform)



data.to_csv("../Data_Set/Final/test_predict.csv",index = False, header = True)


