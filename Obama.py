# -*- coding: utf-8 -*-
"""
Created on Sun May 27 03:10:45 2018

@author: Yusra
"""
import tweepy
from tweepy import OAuthHandler
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
#when using notebook make sure to add
%matplotlib inline  
from textblob import TextBlob

consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = 	''
def  twitter_Access():
     auth = OAuthHandler(consumer_key, consumer_secret)
     auth.set_access_token(access_token, access_secret)
     api = tweepy.API(auth)
     return api
extract = twitter_Access()


#Obamas Analysis
Otweets = extract.user_timeline(screen_name="BarackObama", count=200, tweet_mode='extended')
for tweet in Otweets[:5]:
    print(tweet.full_text)
    print()


Obama = pd.DataFrame(data=[tweet.full_text for tweet in Otweets], columns =['Tweets'])
Obama['Tweet_Length']  = np.array([len(tweet.full_text) for tweet in Otweets])
Obama['ID']   = np.array([tweet.id for tweet in Otweets])
Obama['Date_Posted'] = np.array([tweet.created_at for tweet in Otweets])
Obama['OS'] = np.array([tweet.source for tweet in Otweets])
Obama['Likes']  = np.array([tweet.favorite_count for tweet in Otweets])
Obama['Retweets']    = np.array([tweet.retweet_count for tweet in Otweets])
Obama['Weekday']  = Obama['Date_Posted'].dt.weekday_name
Obama['wt'] =Obama['Date_Posted'].dt.weekday
Obama['Month'] =Obama['Date_Posted'].dt.month
Obama.describe()

Obama = Obama.reset_index()

Obama['Tweets'] = Obama.Tweets.str.replace(r'http\S+', '').str.replace(r'@\S+', '').str.replace('&amp','').str.rstrip()
Obama['Tweet_Length'] = Obama['Tweets'].apply(len)
Obama.loc[(Obama.Tweet_Length > 280), 'Tweets'].count().any()

ObamaStatistics = Obama.describe()

Obama.loc[(Obama.Likes == 0), ['Date_Posted', 'Tweets']].count().sum()
Obama.loc[(Obama.Likes == 0), ['Date_Posted', 'Tweets']]

Obama = Obama[Obama.Likes != 0]
ObamaStatistics = Obama.describe()
Obama.to_csv('Obama.csv')


#data exploration
Obama.loc[Obama['Date_Posted'].idxmin()]
Obama.loc[Obama['Date_Posted'].idxmax()]
Obama.loc[Obama['Likes'].idxmax()]

#mentions of other politicians
Obama.Tweets[Obama.Tweets.str.contains('Trump','Donald', flags= re.IGNORECASE)].count()
Obama.Tweets[Obama.Tweets.str.contains('Hillary','Clinton', flags= re.IGNORECASE)].count()
Obama.Tweets[Obama.Tweets.str.contains('Bernie','Sandars', flags= re.IGNORECASE)].count()
#all amounted to 0


#timeseries and graphs
Daily_Tweets = Obama.groupby(['wt','Weekday']).agg({'Weekday': np.count_nonzero}).plot( colormap="seismic", fontsize=12, kind = 'bar')
Daily_Tweets.set_ylabel('Count of tweets', fontsize=12)
Daily_Tweets.set_xlabel('', fontsize=14)
plt.xticks(np.arange(7), ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
plt.legend(['Count of Tweets']);
           
AvgLikes  = Obama.groupby(['wt']).agg({'Likes': np.mean}).plot(colormap='PiYG', fontsize=12, kind = 'bar')
AvgLikes.set_ylabel('Avearge Likes on Tweets', fontsize=14)
AvgLikes.set_title('Obama')
AvgLikes.set_xlabel('')
plt.xticks(np.arange(7), ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))


TTfavO = pd.Series(data=Obama['Likes'].values, index=Obama['Date_Posted'])
TTretO = pd.Series(data=Obama['Retweets'].values, index=Obama['Date_Posted'])
TimeSeries = TTfavO.plot(colormap="coolwarm_r", figsize=(12,4), label="Favourites", legend=True)
TimeSeries = TTretO.plot(colormap="copper_r", figsize=(12,4), label="Retweets", legend=True);
plt.title('Obama')
plt.xlabel('')


#assigning subjectivity and polarity for sentiment analysis
Obama[['polarity', 'subjectivity']] = Obama['Tweets'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

#finding count for polarity figures
Obama.loc[(Obama.polarity > 0), 'Tweets'].count()
Obama.loc[(Obama.polarity == 0), 'Tweets'].count()
Obama.loc[(Obama.polarity < 0), 'Tweets'].count()

#passing polarity to pie chart
labels = ['Positive', 'Negative', 'Neutral']
sizes = [90, 65, 16]
explode = (0, 0, 0.1)  # explode 1st slice
colors = ['cyan', 'r', 'orchid']
#pie chart
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Obama')

#sample negative tweets
Obama.loc[(Obama.polarity < 0), 'Tweets'].sample(n=4)

