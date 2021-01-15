#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 01:07:30 2020

@author: sqzhao
"""
import os
path = '/Users/sqzhao/Documents/fall20/qmss/Twitter-NLP/lib'
os.chdir(path)

import nltk
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pylab as pl
import pickle

dataset = pd.read_csv('../data/sentiment140/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1',header=None)
X=dataset.iloc[:,5].values
X=pd.Series(X)
y=dataset.iloc[:,0].values

for row in range(0,1600000):
    if y[row]==4:
        y[row]=1
    else:
        y[row]=0




X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=5)


def preprocess(tweet):
    
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    
    #Convert @username to __USERHANDLE
    tweet = re.sub('@[^\s]+','__USERHANDLE',tweet)  
    #tweet = re.sub('@[^\s]+',' ',tweet) 
    
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



X_train = [stem(preprocess(tweet)) for tweet in X_train]
X_test = [stem(preprocess(tweet)) for tweet in X_test]

vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))
X_train_vec = vec.fit_transform(X_train)
nb = MultinomialNB()
nb.fit(X_train_vec,y_train)
X_test_vec = vec.transform(X_test)
pred = nb.predict(X_test_vec)
fitted = nb.predict(X_train_vec)

print(metrics.accuracy_score(y_test, pred)) # multi-class

print(metrics.confusion_matrix(y_test, pred))

cm = metrics.confusion_matrix(y_test, pred)
pl.matshow(cm)
#pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()

print(metrics.classification_report(y_test, pred))


#Save NaiveBayes classifier and vectorizer
clf_NB = open('../output/NaiveBayes.pickle', 'wb')
pickle.dump(nb, clf_NB)
clf_NB.close()

vec_tfidf = open('../output/tfidfVec.pickle','wb')
pickle.dump(vec,vec_tfidf)
vec_tfidf.close()

