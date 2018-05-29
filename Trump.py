# -*- coding: utf-8 -*-
"""
Created on Thu May 17 06:05:45 2018

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

#setting pandas to print full tweets in console
pd.set_option('display.max_colwidth', -1)


# extracting tweets from a user_timeline
Dtweets = extract.user_timeline(screen_name="realDonaldTrump", count=200, tweet_mode='extended', truncated ='False')
for tweet in Dtweets[:5]:
        print(tweet.full_text)
        print()
    
#Information stored in a tweet
print(dir(Dtweets[0]))

#creating dataframe
Donald = pd.DataFrame(data=[tweet.full_text for tweet in Dtweets], columns=['Tweets'])
Donald['Tweet_Length']  = np.array([len(tweet.full_text) for tweet in Dtweets])
Donald['ID']   = np.array([tweet.id for tweet in Dtweets])
Donald['Date_Posted'] = np.array([tweet.created_at for tweet in Dtweets])
Donald['OS'] = np.array([tweet.source for tweet in Dtweets])
Donald['Likes']  = np.array([tweet.favorite_count for tweet in Dtweets])
Donald['Retweets']    = np.array([tweet.retweet_count for tweet in Dtweets])

#using dt to access weekday and month from the Date the tweet was Posted
Donald['Weekday']  = Donald['Date_Posted'].dt.weekday_name
Donald['wt'] =Donald['Date_Posted'].dt.weekday
Donald['Month'] =Donald['Date_Posted'].dt.month

#resetting index
Donald = Donald.reset_index()
#data types in dataframe
Donald.dtypes
#multi-dimensional boolean array for nulls 
Donald.isnull()
#nulls if any in entire database, result boolean
Donald.isnull().values.any()
#nulls in columns of database, result boolean
Donald.isnull().any()
#count of nulls in each column, result int
Donald.isnull().sum()

#descriptive statistics
DonaldStatistics = Donald.describe()
DonaldStatistics.to_csv('DonaldStatistics.csv')

#drilling down tweets containing more than 280 characters
Donald.loc[(Donald.Tweet_Length > 280), 'Tweets'].count()
Donald.loc[(Donald.Tweet_Length > 280), 'Tweets'].sample(n=4)

#cleaning tweet data, using string methods with pandas
Donald['Tweets'] = Donald.Tweets.str.replace(r'http\S+', '').str.replace(r'@\S+', '').str.replace('&amp','').str.rstrip()
Donald['Tweet_Length'] = Donald['Tweets'].apply(len)
Donald.loc[(Donald.Tweet_Length > 280), 'Tweets'].count().any()

#drilling down data where likes are zero
Donald.loc[(Donald.Likes == 0), ['Date_Posted', 'Tweets']].count().sum()
Donald.loc[(Donald.Likes == 0), ['Date_Posted', 'Tweets']]

#removing retweets
Donald = Donald[Donald.Likes != 0]
#descriptive statistics on Donald
DonaldStatistics = Donald.describe()

#saving our cleaned data frame
Donald.to_csv('Donald.csv')

#latest and oldest date, most liked tweets with loc
Donald.loc[Donald['Date_Posted'].idxmin()]
Donald.loc[Donald['Date_Posted'].idxmax()]
Donald.loc[Donald['Likes'].idxmax()]

#using string methods with pandas
#find notable politicians in donalds tweets
Donald.Tweets[Donald.Tweets.str.contains('Obama','Barack', flags= re.IGNORECASE)].count()
Donald.Tweets[Donald.Tweets.str.contains('Hillary','Clinton', flags= re.IGNORECASE)].count()
Donald.Tweets[Donald.Tweets.str.contains('Bernie','Sandars', flags= re.IGNORECASE)].count()
#creating pie chart for politicians mentioned
labels = 'Obama', 'Hillary', 'Bernie'
sizes = [7, 5, 1]
explode = (0.1, 0, 0)  # explode 1st slice
colors = ['gold', 'lightcoral', 'lightskyblue']
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

#count of tweets tweeted on specific weekdays
Donald['Weekday'].value_counts()
Donald = Donald.sort(['wt'])

#Bar chart of aggregated count of tweets on specific weekday
Weekday_Tweets = Donald.groupby(['wt', 'Weekday']).agg({'Weekday': np.count_nonzero}).plot(colormap='seismic', fontsize=12, kind = 'bar')
Weekday_Tweets.set_ylabel('Count of tweets', fontsize=14)
Weekday_Tweets.set_xlabel('', fontsize=14)
plt.xticks(np.arange(7), ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
plt.legend(['Count of Tweets']);

#Bar Chart average likes per weekday
AvgLikes = Donald.groupby(['wt']).agg({'Likes': np.mean}).plot(colormap='PiYG', fontsize=12, kind = 'bar')
AvgLikes .set_ylabel('Avearge Likes on Tweets', fontsize=14)
AvgLikes.set_title('Trump')
AvgLikes.set_xlabel('')
plt.xticks(np.arange(7), ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

#Time Series, creating time series objects
TTfavD = pd.Series(data=Donald['Likes'].values, index=Donald['Date_Posted'])
TTretD = pd.Series(data=Donald['Retweets'].values, index=Donald['Date_Posted'])
#plotting time seres
TTfavD.plot(colormap="coolwarm_r", figsize=(16,4), label="Favourites", legend=True)
TTretD.plot(colormap="copper_r", figsize=(16,4), label="Retweets", legend=True)
plt.title('Donald')
plt.xlabel('')

#assigning subjectivity and polarity for sentiment analysis
Donald[['polarity', 'subjectivity']] = Donald['Tweets'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

#finding count for polarity figures
Donald.loc[(Donald.polarity > 0), 'Tweets'].count()
Donald.loc[(Donald.polarity == 0), 'Tweets'].count()
Donald.loc[(Donald.polarity < 0), 'Tweets'].count()

#passing polarity to pie chart
labels = ['Positive', 'Negative', 'Neutral']
sizes = [122, 45, 22]
explode = (0, 0, 0.1)  # explode 1st slice
colors = ['cyan', 'r', 'orchid']
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Donald')

Donald.loc[(Donald.polarity < 0), 'Tweets'].sample(n=4)








