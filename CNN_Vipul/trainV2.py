import pandas as pd
import numpy as np
import pickle

df_train=pd.read_csv("data/processedData.csv",index_col=False)
labels=df_train[['784']]

df_train.drop(df_train.columns[[784]],axis=1,inplace=True)
df_train.head()