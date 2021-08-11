# Importing Libraries
import time
import pandas as pd
import numpy as np


import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt

# install required libraries

print('data')

nltk.download('vader_lexicon')
nltk.download('stopwords')

##Set seeds for the reproducability
from numpy.random import seed
seed(1)
import tensorflow
from tensorflow.python.framework.random_seed import set_random_seed
set_random_seed(2)
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

tweets_data = pd.read_csv('./Bitcoin_tweets.csv')

# Check data summary
tweets_data.info()

# Checking data shape
tweets_data.shape

# Filtering the data with required fields
tweets_data1 = tweets_data[['date','text']]

#Null values checking
tweets_data1.isnull().sum()
# text Cleasing
stops = nltk.corpus.stopwords.words("english")

def text_preprocess(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    text = re.sub(r'\w*\d+\w*', '', text)# getting rid of alpha numeric
    text = re.sub(r'[^a-zA-Z\s]','',text,re.I|re.A)##removing non letters
    text = re.sub('([#])|([^a-zA-Z])',' ',text)# removing the hashtag
    text = re.sub('<.*>', ' ', text) # removing the html tags
    text = re.sub('[^a-zA-Z\s]+',' ',text) # removing the punctuation
    text = re.sub('[ ]{2,}',' ',text) # removing the extra white space
    txtpost = text.split()
    # remove stopwords
    txtpost = [i for i in txtpost if i not in stops]

    tokens = " ".join(txtpost)
    return tokens

# Above function cleanses the text by removing stopwords,unwanted text,symbols,punctuations,hashtags and extra white space.
# Clean data
tweets_data1['text_new'] = tweets_data1['text'].apply(lambda x: text_preprocess(x))
import matplotlib.pyplot as plt

tweets_data2 = tweets_data1.sort_values(by=['date'])
##VADAR implementation

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
### Getting polarity scores via Vader

result = tweets_data2['text_new'].apply(lambda x: sia.polarity_scores(x))

# List transformation
result1 = list(result)
vader_sentiment = pd.DataFrame.from_records(result1)
tweets_data3 =pd.concat([tweets_data2,vader_sentiment],axis=1)
# Filtering and backing up of data
tweets_data4 = tweets_data3[['date','text_new','compound','neg','pos','neu']]
##Sort values by by date
tweets_data4 = tweets_data4.sort_values(by='date')

Bitfinex_BTCUSD_1h = pd.read_csv('./Bitfinex_BTCUSD_1h.csv')
Bitstamp_BTCUSD_1h = pd.read_csv('./Bitstamp_BTCUSD_1h.csv')
gemini_BTCUSD_1hr = pd.read_csv('./gemini_BTCUSD_1hr.csv')

##Renaming date
gemini_BTCUSD_1hr.loc[:,'date']=gemini_BTCUSD_1hr.Date

##drop old date column
gemini_BTCUSD_1hr = gemini_BTCUSD_1hr.drop('Date',axis=1)

##sort values by date
Bitfinex_BTCUSD_1h = Bitfinex_BTCUSD_1h.sort_values(by=['date'])
Bitstamp_BTCUSD_1h = Bitstamp_BTCUSD_1h.sort_values(by=['date'])
gemini_BTCUSD_1hr = gemini_BTCUSD_1hr.sort_values(by=['date'])
##converting date to datetime format
Bitfinex_BTCUSD_1h['date'] = pd.to_datetime(Bitfinex_BTCUSD_1h['date'])
Bitstamp_BTCUSD_1h['date'] = pd.to_datetime(Bitstamp_BTCUSD_1h['date'])
gemini_BTCUSD_1hr['date'] = pd.to_datetime(gemini_BTCUSD_1hr['date'])

# Setting the date as index for subsetting
gemini_BTCUSD_1hr = gemini_BTCUSD_1hr.set_index(['date'],drop=False)
Bitfinex_BTCUSD_1h = Bitfinex_BTCUSD_1h.set_index(['date'],drop=False)
Bitstamp_BTCUSD_1h = Bitstamp_BTCUSD_1h.set_index(['date'],drop=False)

# Subsetting into date between 2018-05-15 06:00:00' : '2021-07-31 00:00:00 i.e of same range
gemini_BTCUSD_1hr1 = gemini_BTCUSD_1hr.loc['2018-05-15 06:00:00' : '2021-07-31 00:00:00' ]
Bitfinex_BTCUSD_1h1 = Bitfinex_BTCUSD_1h.loc['2018-05-15 06:00:00' : '2021-07-31 00:00:00' ]
Bitstamp_BTCUSD_1h1 = Bitstamp_BTCUSD_1h.loc['2018-05-15 06:00:00' : '2021-07-31 00:00:00' ]

gemini_BTCUSD_1hr1 = gemini_BTCUSD_1hr1.reset_index(drop =True)
Bitfinex_BTCUSD_1h1 = Bitfinex_BTCUSD_1h1.reset_index(drop =True)
Bitfinex_BTCUSD_1h1 = Bitfinex_BTCUSD_1h1.reset_index(drop =True)

## Rename axis
Bitstamp_BTCUSD_1h1 = Bitstamp_BTCUSD_1h1.rename_axis(None)
Bitfinex_BTCUSD_1h1 = Bitfinex_BTCUSD_1h1.rename_axis(None)
gemini_BTCUSD_1hr1 = gemini_BTCUSD_1hr1.rename_axis(None)


## Merging of 3 bitcoin datasets
merged = Bitstamp_BTCUSD_1h1.merge(Bitfinex_BTCUSD_1h1,on='date').merge(gemini_BTCUSD_1hr1,on='date')

## Aggregating the mean (variable wise) of three attributes

merged['avg_close'] = merged[['close_x', 'close_y', 'Close']].mean(axis =1)
merged['avg_open'] = merged[['open_x', 'open_y', 'Open']].mean(axis =1)
merged['avg_high'] = merged[['high_x', 'high_y', 'High']].mean(axis =1)
merged['avg_low'] = merged[['low_x', 'low_y','Low']].mean(axis =1)
merged['avg_volume'] = merged[['Volume USD_x', 'Volume USD_y', 'Volume']].mean(axis =1)
sampledf = merged[['date', 'avg_close', 'avg_open', 'avg_high', 'avg_low', 'avg_volume']]

##convert date to datetime
tweets_data4['date'] = pd.to_datetime(tweets_data4['date'])

#set date as index
tweets_data4 = tweets_data4.set_index(['date'],drop=False)

## Subset tweets to dates
tweets_data5 = tweets_data4.loc['2018-05-15 06:00:00' : '2021-07-31 00:00:00' ]

##reset index to drop
tweets_data5 = tweets_data5.reset_index(drop =True)
tweets_data5 = tweets_data5.rename_axis(None)
## sort values by date
tweets_data6 = tweets_data5.sort_values(by ='date',ascending=False)

## check whether string in date ?
tweets_data6[tweets_data6['date'].map(type) == str]

##Min max normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

tweets_data6['compound_norm'] = minmax_scale(tweets_data6['compound'].astype(np.float64))
tweets_data6 = tweets_data6.set_index(["date"],drop =False)
##Getting Aggregates of polarity data to hourly by resample
tweets_data_updtd = tweets_data6.resample('H').agg(dict(compound= 'mean',compound_norm='mean',neg='mean',pos='mean',neu='mean')).ffill()

##Re assign date as column from date index
tweets_data_updtd['date']=tweets_data_updtd.index

##reset index by dropping it
tweets_data_updtd = tweets_data_updtd.reset_index(drop = True)

##Make a copy of tweets polarity data
twt_pol = tweets_data_updtd.copy()

## Checking missing data
tweets_data_updtd.isna().sum()/len(tweets_data_updtd)
##Resetting index
sampledf1 = sampledf.set_index('date',drop= False)
##fixing range of dates
sampledf1 = sampledf1.loc['2021-02-05 10:00:00' : '2021-03-12 23:00:00' ]

#reset the dates for merging
sampledf2 = sampledf1.rename_axis(None)
twt_pol1 = twt_pol.rename_axis(None)
sampledf5 = sampledf2.copy()
sampledf5 = sampledf5.reset_index(drop=True)
sampledf5 = sampledf5.rename_axis(None)

twt_pol1 = twt_pol1.copy()
twt_pol1 = twt_pol1.reset_index(drop=True)
twt_pol1 = twt_pol1.rename_axis(None)

##merging Tweets data and price data
tweet_price = sampledf5.merge(twt_pol1,on='date')
from statsmodels.tsa.arima.model import ARIMA

from sklearn.model_selection import train_test_split

X = tweet_price['avg_close'].values
train = X[0:800]# train data
test = X[800:]  #test data
predictions = []

steps = len(tweet_price) - len(train)
print(steps)

import statsmodels.api as sm
import matplotlib
from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
## knowing the Trend ,seasonality,randomness and seasonality
decomposition = sm.tsa.seasonal_decompose(tweet_price['avg_close'].values, model='additive',period = 24)
fig = decomposition.plot()
plt.show()

ts_data = tweet_price.copy()
#DataFrame.shift(periods=1, freq=None, axis=0, fill_value=<no_default>)[source]
## Adding change in the lag as input
ts_data['diffS']=ts_data['avg_close'].diff()
ts_data['lag']=ts_data['diffS'].shift()
ts_data.dropna(inplace=True)

#ts_data.index.freq = 'H'
ts_data = ts_data.set_index(['date'],drop=False)
ts_data.index = pd.DatetimeIndex(ts_data.index.values,
                               freq=ts_data.index.inferred_freq)

#X = ts_data['avg_close']
train = ts_data.iloc[0:800,:]# train data
test = ts_data.iloc[800:,:]     ##test data
predictions = []

## Making these varibales as exogenous variables for sarimax model
ts_data['diffavg_open']=ts_data['avg_open'].diff()
ts_data['lag_avg_open']=ts_data['diffavg_open'].shift()
ts_data['diffavg_high']=ts_data['avg_high'].diff()
ts_data['lag_avg_high']=ts_data['diffavg_high'].shift()
ts_data['diffavg_low']=ts_data['avg_low'].diff()
ts_data['lag_diffavg_low']=ts_data['diffavg_low'].shift()
ts_data['diffavg_volume']=ts_data['avg_volume'].diff()
ts_data['lag_diffavg_volume']=ts_data['diffavg_volume'].shift()

ts_data.dropna(inplace=True)

train = ts_data.iloc[0:800,:]# train data
test = ts_data.iloc[800:,:]   ## test data
predictions = []

from statsmodels.tsa.statespace.sarimax import SARIMAX

model3=SARIMAX(endog=train['avg_close'],exog=train[['lag','lag_avg_open','lag_avg_high','lag_diffavg_low','lag_diffavg_volume']],order=(2,1,2))
results3=model3.fit()
##print(results3.summary())

import pmdarima as pm

# SARIMAX Model
start_time = time.time()

sxmodel = pm.auto_arima(train['avg_close'], exogenous=train[['lag','lag_avg_open','lag_avg_high','lag_diffavg_low','lag_diffavg_volume','compound']],
                           start_p = 1,start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

sxmodel.summary()
end_time = time.time()
execution_time = end_time-start_time
print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(execution_time)))
# Future steps to predict
steps = len(ts_data) - len(train)
print(steps)

# Getting predictions
predictions = sxmodel.predict(n_periods=steps,exogenous = test[['lag','lag_avg_open','lag_avg_high','lag_diffavg_low','lag_diffavg_volume','compound']])
#= results2.forecast(steps=steps,exog=test['lag'])

actual = test['avg_close'].values
from sklearn.metrics import mean_squared_error
test_set_rmse = (np.sqrt(mean_squared_error(actual, predictions)))
print(test_set_rmse)

