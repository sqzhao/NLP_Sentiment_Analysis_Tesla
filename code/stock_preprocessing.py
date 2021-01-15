#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:42:06 2020

@author: sqzhao
"""
import os
path = '/Users/sqzhao/Documents/fall20/qmss/Twitter-NLP/lib'
os.chdir(path)

import pandas as pd

#nasdaq composite
nasdaq = pd.read_csv('../data/Nasdaq.csv')
nasdaq = nasdaq[["Date", "Close"]]
nasdaq = nasdaq.rename(columns = {'Close': "Nasdaq"})

nasdaq["Nasdaq_1d"] = nasdaq.Nasdaq.shift(1)
nasdaq["Nasdaq_2d"] = nasdaq.Nasdaq.shift(2)
nasdaq["Nasdaq_3d"] = nasdaq.Nasdaq.shift(3)

nasdaq = nasdaq.fillna(method='backfill') 

# tesla stock
tesla_stock = pd.read_csv('../data/TSLA_stock2020.csv')
tesla_stock = tesla_stock[["Date", "Volume","Close"]]
tesla_stock = tesla_stock.rename(columns = {'Close': "Price"})

tesla_stock["Price_1d"] = tesla_stock.Price.shift(1)
tesla_stock["Price_2d"] = tesla_stock.Price.shift(2)
tesla_stock["Price_3d"] = tesla_stock.Price.shift(3)
tesla_stock = tesla_stock.fillna(method='backfill') 

# combine
stock = tesla_stock.merge(nasdaq, on = 'Date')
stock['Date']= pd.to_datetime(stock['Date'],format = '%Y-%m-%d').dt.date

#stock.to_csv("../data/stock.csv")
#stock = pd.read_csv('../data/stock.csv')












#### stock data ------------------------------------------------------------------

#### preprocess ------------------------------------------------------------------
nasdaq = pd.read_csv('../data/Nasdaq.csv')
nasdaq = nasdaq[["Date", "Close"]]
nasdaq = nasdaq.rename(columns = {'Close': "index"})
tesla_stock_raw = pd.read_csv('../data/TSLA_stock2020.csv')
tesla_stock_raw = tesla_stock_raw[["Date", "Close", "Volume"]]
# tesla_stock.dtypes
# tesla_stock.info()
tesla_stock = tesla_stock_raw.merge(nasdaq, on = 'Date')

tesla_stock['Date']= pd.to_datetime(tesla_stock['Date'],format = '%Y-%m-%d').dt.date
