#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:23:04 2020

@author: sqzhao
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn import linear_model
from sklearn import metrics


### Setting working directory
path = '/Users/sqzhao/Documents/fall20/qmss/Twitter-NLP/lib'
os.chdir(path)

combine = pd.read_csv('../data/combine.csv', index_col=0)
combine.head(5)
combine.columns

train_start_index = 0
train_end_index = int(len(combine)*(80/100)) - 1 #154
test_start_index = int(len(combine)*(80/100))
test_end_index = len(combine) #194

split_point = int(len(combine)*(80/100)) #224

train = combine[:split_point]
test = combine[split_point:]

train_y = train[["Price"]]
train_x = train[['negative', 'neutral', 'Price_1d','Nasdaq']] # feature selection using correlation matrix
               # 'positive', 
                # 'Volume', 'Price_1d', 'Price_2d', 'Price_3d', 
               # 'Nasdaq', 'Nasdaq_1d', 'Nasdaq_2d', 'Nasdaq_3d']]
test_y = test[["Price"]]
test_x = test[['negative', 'neutral', 'Price_1d','Nasdaq']]

## Random Forest --------------------------------------------------------------
rf = RandomForestRegressor(random_state=0)
rf.fit(train_x, train_y) # should fit with grid search
pred_y = rf.predict(test_x)
fitted_y = rf.predict(train_x)

# MSE
print("train MSE for RF:",metrics.mean_squared_error(train_y,fitted_y))
print("test MSE for RF:",metrics.mean_squared_error(test_y, pred_y))

# plot
idx = np.arange(int(test_start_index),int(test_end_index))
df = pd.DataFrame({'pred_y': pred_y, 'fitted_y': test_y.Price})
ax = df.plot()
ax.set_xlabel("Day Index")
ax.set_ylabel("Stock Prices")
ax.set_title('Random Forest', fontsize=15)

# print(idx)
pred_df = pd.DataFrame(data=pred_y, index = idx, columns=['Prices'])

ax = pred_df.rename(columns={"Prices": "predicted_price"}).plot(title='predicted prices')#predicted value
ax.set_xlabel("Indexes")
ax.set_ylabel("Stock Prices")
fig = test_y.rename(columns={"Prices": "actual_price"}).plot(ax = ax).get_figure()#actual value
fig.savefig("random forest.png")



# Linear Regression------------------------------------------------------------
lr = LinearRegression()
#lr = linear_model.Lasso(alpha=0.5)
lr.fit(train_x, train_y)
lr.coef_
pred_y = lr.predict(test_x)
fitted_y = lr.predict(train_x)

# MSE
print("train MSE for LR:",metrics.mean_squared_error(train_y,fitted_y))
print("test MSE for LR:",metrics.mean_squared_error(test_y, pred_y))

# plot
idx = np.arange(int(test_start_index),int(test_end_index))
df = pd.DataFrame({'pred_y': pred_y.reshape(-1), 'fitted_y': test_y.Price})
ax = df.plot()
ax.set_xlabel("Day Index")
ax.set_ylabel("Stock Prices")
ax.set_title('Linear Regression', fontsize=15)


# MLP: multilayer perceptron---------------------------------------------------
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)
mlp = MLPRegressor(hidden_layer_sizes=(60,60), max_iter=150000000, solver='adam')
mlp.fit(train_x, train_y)
pred_y = mlp.predict(test_x)
fitted_y = mlp.predict(train_x)

# MSE
print("train MSE for MLP:",metrics.mean_squared_error(train_y,fitted_y))
print("test MSE for MLP:",metrics.mean_squared_error(test_y, pred_y))

# plot
idx = np.arange(int(test_start_index),int(test_end_index))
df = pd.DataFrame({'pred_y': pred_y.reshape(-1), 'fitted_y': test_y.Price})
ax = df.plot()
ax.set_xlabel("Day Index")
ax.set_ylabel("Stock Prices")
ax.set_title('Multilayer Perceptron', fontsize=15)


#### correlation matrix for feature selection----------------------------------
corr = train.corr()
# Generate a mask for the upper triangle
#mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

ax = plt.axes()
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,  cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)#mask=mask,)

ax.set_title('Feature Correlation Matrix')
plt.show()






#### Naive Bayes:--------------------------------------------------------------
combine_nb = pd.read_csv("../data/combine_nb.csv", index_col=0)
combine = combine_nb


train_start_index = 0
train_end_index = int(len(combine)*(80/100)) - 1 #154
test_start_index = int(len(combine)*(80/100))
test_end_index = len(combine) #194

split_point = int(len(combine)*(80/100)) #224

train = combine[:split_point]
test = combine[split_point:]

train_y = train[["Price"]]
#train_x = train[['neg_per', 'Price_1d' ,'Nasdaq']] # feature selection using correlation matrix
train_x = train[['Price_1d' ,'Nasdaq']] 
               # 'positive', 
                # 'Volume', 'Price_1d', 'Price_2d', 'Price_3d', 
               # 'Nasdaq', 'Nasdaq_1d', 'Nasdaq_2d', 'Nasdaq_3d']]
test_y = test[["Price"]]
#test_x = test[['neg_per', 'Price_1d' ,'Nasdaq']]
test_x = test[['Price_1d' ,'Nasdaq']]



