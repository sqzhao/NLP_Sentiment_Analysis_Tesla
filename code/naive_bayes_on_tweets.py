#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 04:47:22 2020

@author: sqzhao
"""
import os
path = '/Users/sqzhao/Documents/fall20/qmss/Twitter-NLP/lib'
os.chdir(path)

import pickle
#from nb.py import preprocess, stem
def preprocess(tweet):
    
    #Convert www.* or https?://* to URL
    #tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    
    #Convert @username to __USERHANDLE
    #tweet = re.sub('@[^\s]+','__USERHANDLE',tweet)  
    tweet = re.sub('@[^\s]+',' ',tweet) 
    
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    
    #trim
    tweet = tweet.strip('\'"')
    
    # Repeating words like hellloooo
    repeat_char = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    tweet = repeat_char.sub(r"\1\1", tweet)
    
    #Emoticons
    emoticons = \
    [
     ('__positive__',[ ':-)', ':)', '(:', '(-:', \
                       ':-D', ':D', 'X-D', 'XD', 'xD', \
                       '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ] ),\
     ('__negative__', [':-(', ':(', '(:', '(-:', ':,(',\
                       ':\'(', ':"(', ':((','D:' ] ),\
    ]

    def replace_parenthesis(arr):
       return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]
    
    def join_parenthesis(arr):
        return '(' + '|'.join( arr ) + ')'

    emoticons_regex = [ (repl, re.compile(join_parenthesis(replace_parenthesis(regx))) ) \
            for (repl, regx) in emoticons ]
    
    for (repl, regx) in emoticons_regex :
        tweet = re.sub(regx, ' '+repl+' ', tweet)

     #Convert to lower case
    tweet = tweet.lower()
    
    return tweet

# Tokenizing and Stemming of Tweets
def stem(tweet):
        stemmer = nltk.stem.PorterStemmer()
        tweet_stem = ''
        words = [word if(word[0:2]=='__') else word.lower() \
                    for word in tweet.split() \
                    if len(word) >= 3]
        words = [stemmer.stem(w) for w in words] 
        tweet_stem = ' '.join(words)
        return tweet_stem


#------------------------------------------------------------------------------
clf_NB_f = open('../output/NaiveBayes.pickle',mode="rb")
nb = pickle.load(clf_NB_f)
clf_NB_f.close()

vec_tfidf = open('../output/tfidfVec.pickle',mode="rb")
vec = pickle.load(vec_tfidf)
vec_tfidf.close()
###

tidy_tesla_tweets = pd.read_csv('../data/tesla_tweets_clean.csv', index_col=0)
del tidy_tesla_tweets["Unnamed: 0.1"]

for i in tidy_tesla_tweets.text:
    print(type(i))
pred_tweets = stem(preprocess(tidy_tesla_tweets.iloc[0,1]))


X_test = [stem(preprocess(tweet)) for tweet in tidy_tesla_tweets.text]

X_test_vec = vec.transform(X_test)
pred = nb.predict(X_test_vec)
tidy_tesla_tweets["sentiment"] = pred

senti_all = tidy_tesla_tweets[['Date', 'sentiment']]
df = senti_all.groupby(['Date', 'sentiment']).size().to_frame().reset_index().rename(columns= {0: 'cnt'})
# pivot_wider: https://www.datasciencemadesimple.com/reshape-long-wide-pandas-python-pivot-function/
df2 = df.pivot(index='Date', columns='sentiment', values='cnt').fillna(0).rename(columns= {0: 'neg', 1:'pos'})

df2['neg_per'] = df2['neg']/df2['pos']

senti_perc = df2[["neg_per"]].reset_index()
#senti_perc.to_csv("../data/senti_perc.csv")

stock = pd.read_csv('../data/stock.csv',index_col=0)
combine_nb = stock.merge(senti_perc, on = 'Date', how='left').dropna(axis=0,how='any')
combine_nb.columns
#combine_nb.to_csv("../data/combine_nb.csv")




