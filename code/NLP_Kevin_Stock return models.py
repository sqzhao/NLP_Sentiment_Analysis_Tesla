#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[2]:


df = pd.read_csv("combine.csv",index_col=0)


# In[3]:


df = df[["Volume","Price","Nasdaq","compound","positive","negative","neutral"]] # get the feature columns that will be used


# In[4]:


df['return'] = df['Price'].pct_change().shift(1) # shift return to use pre_day info as features, to avoid looking ahead bias
df['Nasdaq'] = df['Nasdaq'].pct_change()
df.drop(["Price"],axis=1,inplace=True)
df.dropna(inplace=True)


# In[5]:


## four kinds of feature sets
base_feature_set = ["Volume","Nasdaq"]
vader_feaetue_set = ["compound"]+base_feature_set
nb_feature_set = ["positive","negative","neutral"]+base_feature_set
all_feature_set = ["positive","negative","neutral"] + ["compound"] + base_feature_set
X_base = df[base_feature_set]
X_vader = df[vader_feaetue_set]
X_nb = df[nb_feature_set]
X_all = df[all_feature_set]
y = df["return"]


# In[6]:


## split train and test
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y,shuffle=False)
X_train_vader, X_test_vader, y_train_vader, y_test_vader = train_test_split(X_vader, y,shuffle=False)
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y,shuffle=False)
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_base, y,shuffle=False)


# In[7]:


def rolling_validation(X,y,model, train_size = 30): # use rolling validation for time series data to avoid looking ahead bias
    n = len(X)
    RMSE = []
    for idx in range(train_size+1, n):
        cur_X, cur_y = X.iloc[idx-train_size-1:idx,], y.iloc[idx-train_size-1:idx,]
        X_train, X_validation, y_train, y_validation = train_test_split(cur_X, cur_y,train_size= train_size,shuffle=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_validation) 
        mse = mean_squared_error(y_validation.values, y_pred, squared=False)
        RMSE.append(mse)
    return np.mean(RMSE)


# In[34]:


def find_best_alpha(X,y):
    n_alphas = 20
    alphas = np.logspace(-5, 1, n_alphas)
    best_rmse = float("inf")
    RMSEs = []
    for alpha in alphas:
        model = Ridge(alpha=alpha,normalize=True)
        rmse = rolling_validation(X,y,model)
        RMSEs.append(rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    print("best alpha is:",best_alpha)
    plt.plot(alphas,RMSEs)
    plt.title("Ridge RMSE")
    plt.xlabel("alpha")
    plt.ylabel("RMSE")
    return best_alpha
best_alpha = find_best_alpha(X_train_base,y_train_base)


# In[8]:


def get_test_mse(X_train,y_train,X_test,y_test,model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 
    rmse = mean_squared_error(y_test.values, y_pred, squared=False)
    return rmse


# In[9]:


def measure_perf(model):
    base_rmse = get_test_mse(X_train_base,y_train_base,X_test_base,y_test_base,model)
    vader_rmse = get_test_mse(X_train_vader,y_train_vader,X_test_vader,y_test_vader,model)
    nb_rmse = get_test_mse(X_train_nb,y_train_nb,X_test_nb,y_test_nb,model)
    all_rmse = get_test_mse(X_train_all,y_train_all,X_test_all,y_test_all,model)
    
    feature_sets = ["Base","VADER","NB","All"]
    RMSEs = [base_rmse,vader_rmse,nb_rmse,all_rmse]

    plt.bar(feature_sets,RMSEs,color=['b', 'm', 'y',"c"])
    for feature_set,rmse in zip(feature_sets,RMSEs):
        plt.text(feature_set, rmse, '%.6f' % rmse, ha='center', va= 'bottom',fontsize=7)
    plt.title("RMSE results of different feature sets");
    plt.xlabel("feature sets")
    plt.ylabel("RMSE")
    plt.style.use("ggplot")
    plt.show()


# In[10]:


plt.style.use("ggplot")


# In[11]:


best_alpha = 2.3357214690901213


# In[12]:


ridge_model = Ridge(alpha=best_alpha,normalize=True)
measure_perf(ridge_model)


# In[40]:


def find_best_depth(X,y):
    depths = list(range(10,101,10))
    best_rmse = float("inf")
    RMSEs = []
    for depth in depths:
        model = RandomForestRegressor(max_depth=depth)
        rmse = rolling_validation(X,y,model)
        RMSEs.append(rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_depth = depth
    print("best max depth is:",best_depth)
    plt.plot(depths,RMSEs)
    plt.title("Random Forest RMSE")
    plt.xlabel("depth")
    plt.ylabel("RMSE")
    return best_depth
best_depth = find_best_depth(X_train_base,y_train_base)


# In[ ]:


def find_best_estimator_num(X,y,best_depth):
    estimator_nums = range(5,200,10)
    best_rmse = float("inf")
    RMSEs = []
    for estimator_num in estimator_nums:
        model = RandomForestRegressor(n_estimators=estimator_num, max_depth=best_depth)
        rmse = rolling_validation(X,y,model)
        RMSEs.append(rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_estimator_num = estimator_num
    print("best estimator num is:",best_estimator_num)
    plt.plot(estimator_nums,RMSEs)
    plt.title("Random Forest RMSE")
    plt.xlabel("estimator nums")
    plt.ylabel("RMSE")
    return best_estimator_num
best_estimator_num = find_best_estimator_num(X_train_base,y_train_base,best_depth)


# In[13]:


best_estimator_num = 115
best_depth = 90
rf_model = RandomForestRegressor(n_estimators=best_estimator_num,max_depth=best_depth)
measure_perf(rf_model)

