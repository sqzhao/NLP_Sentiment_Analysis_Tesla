#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 23:15:52 2020

@author: sqzhao
"""
### pie plot: sentiment per tweet ---------------------------------------------
# sentiment score for each tweet
senti_scores = pd.read_csv('../data/senti_scores.csv', index_col=0)

pos_cnt=0
neg_cnt=0
neu_cnt = 0

for i in range (0,len(senti_scores)):
    get_value=senti_scores.compound[i]
    if(float(get_value)>0):######## dominated sentiment
        pos_cnt +=1 
    elif(float(get_value)<0):  
        neg_cnt +=1
    else:
        neu_cnt +=1

posper=(pos_cnt/(len(senti_scores)))*100
negper=(neg_cnt/(len(senti_scores)))*100
neuper=(neu_cnt/(len(senti_scores)))*100
print("% of positive tweets= ",posper)
print("% of negative tweets= ",negper)
print("% of neutral tweets= ",neuper)
import numpy as np
import matplotlib.pyplot as plt

arr=np.asarray([posper,negper, neuper], dtype=int)
plt.pie(arr,labels=['positive','negative', 'neutral'])
plt.plot()



### word cloud ----------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
### Setting working directory
import os
path = '/Users/sqzhao/Documents/fall20/qmss/Twitter-NLP/lib'
os.chdir(path)


tidy_tesla_tweets = pd.read_csv('../data/tesla_tweets_ordered.csv')
grouped_tweets= pd.read_csv("../data/grouped_tweets.csv", index_col=0).reset_index()

raw_tweets = grouped_tweets[['text']].copy()
raw_string = ' '.join(raw_tweets['text'])

no_links = re.sub(r'http\S+', ' ', raw_string)
no_unicode = re.sub(r"\\[a-z][a-z]?[0-9]+", ' ', no_links)
no_special_characters = re.sub('[^A-Za-z ]+', ' ', no_unicode)
no_special_characters[:10]

words = no_special_characters.split(" ")
words = [w for w in words if len(w) > 2]  # ignore a, an, be, ...
words = [w.lower() for w in words]
words = [w for w in words if w not in STOPWORDS]
words[:10]


clean_string = ' '.join(words)

# def check_en(var):
#     import enchant
#     d = enchant.Dict("en_US")
#     tmp = var.split() #tokenize
#     tmp_list = list()
#     for word in tmp:
#         if d.check(word):
#             tmp_list.append(word)
#     tmp = ' '.join(tmp_list)
#     return tmp
# import enchant
# eng_string = check_en(clean_string)


mask = np.array(Image.open("../figs/logo.jpg"))


wc = WordCloud(collocations=False, background_color="white", max_words=1000,mask=mask)

res = wc.generate(clean_string)

plt.figure(figsize=(50,50))
plt.imshow(res, interpolation='bilinear') 
plt.title('Twitter Generated Cloud', size=20)
plt.axis("off")
#plt.savefig('tweetcloud.png', dpi=300)
plt.show()
