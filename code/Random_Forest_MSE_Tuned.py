#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:07:19 2020

@author: wangzehui
"""

import pandas as pd
df = pd.read_csv("combine.csv",index_col=0)

df['return'] = df['Price'].pct_change()
df['return_1'] = df['Price_1d'].pct_change()
df['return_2'] = df['Price_2d'].pct_change()
df['return_3'] = df['Price_3d'].pct_change()
df.drop(["Price",'Price_1d','Price_2d','Price_3d',"Nasdaq"],axis=1,inplace=True)
df['Nasdaq_1d'] = df['Nasdaq_1d'].pct_change()
df['Nasdaq_2d'] = df['Nasdaq_2d'].pct_change()
df['Nasdaq_3d'] = df['Nasdaq_3d'].pct_change()

df.dropna(inplace=True)
df.drop("Date",axis=1,inplace=True)

print(df.head(5))

from sklearn import datasets, linear_model
X = df[["Volume","Nasdaq_1d","Nasdaq_2d","Nasdaq_3d","compound","positive","negative","neutral","return_1","return_2","return_3"]]
X_n = df[["Volume","Nasdaq_1d","Nasdaq_2d","Nasdaq_3d","return_1","return_2","return_3"]]
y = df["return"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=False, random_state=1)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_n, y,shuffle=False, random_state=1)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
def rolling_val(X_train,y_train,max_depth):
    sample_num = 30
    n = len(X_train)
    RMSE = []
    for sequence in range(sample_num+1, n):
        X_train_small, X_test_small = X_train.iloc[sequence-sample_num-1:sequence-1,],X_train.iloc[sequence-1:sequence,]
        y_train_small, y_test_small = y_train.iloc[sequence-sample_num-1:sequence-1,],y_train.iloc[sequence-1:sequence,]
        mdl = RandomForestRegressor(max_depth)
        mdl.fit(X_train_small, y_train_small)
        y_pred = mdl.predict(X_test_small) 
        error = mean_squared_error(y_test_small.values, y_pred)
        RMSE.append(error)
    return np.mean(RMSE)
rolling_val(X_train,y_train,1)


def find_best_depth(X_train,y_train):
    n_depths = 200
    depths = {1,2,3,4,5,6,7}
    best_mse = float("inf")
    for depth in depths:
        mse = rolling_val(X_train,y_train,depth)
        if mse < best_mse:
            best_mse = mse
            best_depth = depth
    return best_depth, best_mse

def get_testing_mse(X_train,y_train,X_test,y_test,depth):
    mdl = RandomForestRegressor(depth)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test) 
    error = mean_squared_error(y_test.values, y_pred)
    return error


rolling_val(X_train,y_train,2)
get_testing_mse(X_train,y_train,X_test,y_test,2)
find_best_depth(X_train_n,y_train_n)
get_testing_mse(X_train_n,y_train_n,X_test_n,y_test_n,2)
