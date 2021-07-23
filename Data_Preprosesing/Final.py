import pandas as pd
import random as rd
import numpy as np
import category_encoders as ce

## Importing the dataset & reading two csv files
data1 = pd.read_csv('../Data_Set/Final/Marvel_Data.csv',usecols = [2,4,5,6,7,8,10,14])
data2 = pd.read_csv('../Data_Set/Final/DC_Data.csv',usecols = [2,4,5,6,7,8,10,14])

### Combined the dataset
final_data = pd.concat([data1, data2])


x = pd.DataFrame(final_data.iloc[:,2].values)
s = pd.DataFrame(final_data.iloc[:,3].values)
p = pd.DataFrame(final_data.iloc[:,4].values)

### missing data handaling
 
## Characters
while(x.isna().sum().all() != 0):  # to find nan 
    str = np.random.choice(['Good Characters','Neutral Characters','Bad Characters'])
    x.fillna(str,inplace=True,limit=1)
# print (len(x))

## Sex
while(s.isna().sum().all() != 0):  # to find nan 
    str = np.random.choice(['Male Characters','Female Characters'])
    s.fillna(str,inplace=True,limit=1)
# print (len(s))

## Power
while(p.isna().sum().all() != 0):  # to find nan 
    str = np.random.choice(['Secret Identity','Public Identity','No Dual Identity'])
    p.fillna(str,inplace=True,limit=1)
# print (len(p))

## To save & convert dataframe to csv
# final_data.to_csv("Final_data.csv")


###Label Enconding/ Ordinal Encoding

# data = pd.read_csv('final_data.csv')

# #Characters#

C_dat=pd.DataFrame(final_data)

encoder= ce.OrdinalEncoder(cols=['ALIGN'],return_df=True,mapping=[
    {'col':'ALIGN','mapping':{'Good Characters':0,'Neutral Characters':1,
                              'Bad Characters':2,'Reformed Criminals':2}}])

# ##Original data
# C_dat

## fit and transform train data 

C_dat_transformed = encoder.fit_transform(C_dat)

# #Gender#

# G_dat=pd.DataFrame(final_data)

encoder= ce.OrdinalEncoder(cols=['SEX'],return_df=True,mapping=[
    {'col':'SEX','mapping':{'Female Characters':3,'Male Characters':4,'Genderfluid Characters':5,'Agender Characters':6}}])

# ##Original data
# G_dat

## fit and transform train data 

G_C_dat_transformed = encoder.fit_transform(C_dat_transformed)


## To save & convert dataframe to csv
G_C_dat_transformed.to_csv("../Data_Set/Final/test_data.csv",index = False, header = True)






