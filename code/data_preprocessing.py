#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:32:37 2020

@author: sqzhao
"""
### Setting working directory
import os
path = '/Users/sqzhao/Documents/fall20/qmss/Twitter-NLP/lib'
os.chdir(path)


## Twitter data Cleanup -----------------------------------------------------
import pandas as pd
tesla_tweets = pd.read_csv('../data/tesla_tweets.csv',index_col=False)
tidy_tesla_tweets = tesla_tweets[["date_time", "text"]]
tidy_tesla_tweets = tidy_tesla_tweets.dropna(axis=0,how='any')
tidy_tesla_tweets = tidy_tesla_tweets.sort_values(by=['date_time'])[3:]
tidy_tesla_tweets = tidy_tesla_tweets.reset_index()
del tidy_tesla_tweets['index']
tidy_tesla_tweets.columns

tidy_tesla_tweets['date_time'] = pd.to_datetime(tidy_tesla_tweets['date_time'], format = '%Y-%m-%d %H:%M:%S', errors='coerce')

# remove NaN date_time
tidy_tesla_tweets['date_time'] = tidy_tesla_tweets['date_time'].dt.date
tidy_tesla_tweets = tidy_tesla_tweets.rename(columns={"date_time": "Date"})
#tidy_tesla_tweets.to_csv("../data/tesla_tweets_ordered.csv")
#tidy_tesla_tweets = pd.read_csv('../data/tesla_tweets_ordered.csv')

# remove_urls
tidy_tesla_tweets['text'] = tidy_tesla_tweets['text'].str.replace("http.?://[^\s]+[\s]?","", regex = True)
# remove_usernames
tidy_tesla_tweets['text'] = tidy_tesla_tweets['text'].str.replace("@[^\s]+[\s]?","", regex = True)
# remove_hashtags
tidy_tesla_tweets['text'] = tidy_tesla_tweets['text'].str.replace("#","", regex = True)

# remove special characters and unrolls hashtag to text
# for remove in [",", ":", "\"", "=", "&", ";", "%", "$","@", "%", "^", 
#                "*", "(", ")", "{", "}","[", "]", "|", "/", "\\", ">", 
#                "<", "-","?", ".", "'","--", "---", "#"]:
#    tidy_tesla_tweets['text'] = tidy_tesla_tweets['text'].str.replace(remove,"", regex = True)

tidy_tesla_tweets = tidy_tesla_tweets.sort_values(by=['Date']).reset_index()
del tidy_tesla_tweets['index']

####tidy_tesla_tweets.to_csv("../data/tesla_tweets_clean.csv") !!!



## Data Prep for VADER -------------------------------------------------------
grouped_tweets = tidy_tesla_tweets.groupby([tidy_tesla_tweets['Date']]).agg(lambda column: "".join(column))
grouped_tweets = grouped_tweets.dropna(axis=0,how='any')
#grouped_tweets.to_csv("../data/grouped_tweets.csv") !!!
#grouped_tweets = pd.read_csv('../data/grouped_tweets.csv',index_col=0)
grouped_tweets.columns

#### Sentiment Scores for grouped_tweets data of each day:
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

#tmp = tidy_tesla_tweets
tmp = grouped_tweets
grouped_tweets['scores'] = grouped_tweets['text'].apply(lambda tweet_text: sid.polarity_scores(tweet_text))
grouped_tweets.scores
grouped_tweets['compound'] = grouped_tweets['scores'].apply(lambda s: s.get('compound'))
grouped_tweets['positive'] = grouped_tweets['scores'].apply(lambda s: s.get('pos'))
grouped_tweets['negative'] = grouped_tweets['scores'].apply(lambda s: s.get('neg'))
grouped_tweets['neutral'] = grouped_tweets['scores'].apply(lambda s: s.get('neu'))
senti_scores_join = grouped_tweets[['compound', 'positive', 'negative','neutral']]
senti_scores_join = senti_scores_join.reset_index()

#senti_scores_join.to_csv("../data/senti_scores_join.csv") !!!
senti_scores_join = pd.read_csv('../data/senti_scores_join.csv', index_col=0)

##### combine stock data and sentiment scores from grouped tweets every day
stock = pd.read_csv('../data/stock.csv',index_col=0)
combine = stock.merge(senti_scores_join, on = 'Date', how='left').dropna(axis=0,how='any')
combine = combine.dropna(axis=0,how='any')
combine.columns
#combine.to_csv("../data/combine.csv")





### Not in use currently
# def sentiment_analyzer_scores(lb):
#     #score = analyser.polarity_scores(text)
#     #lb = score['compound']
#     if lb >= 0.05:
#         return 1
#     elif (lb > -0.05) and (lb < 0.05):
#         return 0
#     else:
#         return -1
