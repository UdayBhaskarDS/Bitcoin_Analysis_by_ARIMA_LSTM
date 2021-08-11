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

# merging Tweets data and price data
tweet_price = sampledf5.merge(twt_pol1,on='date')

price_dataFrame = tweet_price[['avg_open','avg_high','avg_low','avg_volume','avg_close']]
sent_dataFrame = tweet_price[['compound']]
print(price_dataFrame.shape)
print(sent_dataFrame.shape)


# 2 - Feature Scaling
# don't need to scale sentiment feature since it is already between 0 and 1
sentiment = sent_dataFrame.values.reshape(-1, sent_dataFrame.shape[1])
#print(sentiment.shape)

# price features scaling/standardization
prices = price_dataFrame.values.reshape(-1, price_dataFrame.shape[1])
#print(prices.shape)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)
print(scaled_prices.shape)

# 3 - Spliting train and test sets
train_size = int(len(scaled_prices) * 0.8)
test_size = len(scaled_prices) - train_size
train, test = scaled_prices[0:train_size,:], scaled_prices[train_size:len(scaled_prices),:]
print(f'Train set size: {len(train)}, Test set size: {len(test)}')
split = train_size

# Create function for lag value processing
def create_lag_window_dataset(price_data, sentiment_data, look_back, use_sentiment=True):
    dataX, dataY = [], []
    dataset_range = len(price_data) - look_back
    for i in range(dataset_range):
        if i >= look_back:
            a = price_data[i-look_back:i+1, :]
            a = a.reshape(1, -1)
            if use_sentiment:
                b = sentiment_data[i-look_back:i+1, :]
                b = b.reshape(1, -1)
                a = np.hstack([a, b])
            a = a.tolist()
            dataX.append(a)
            dataY.append(price_data[i+look_back, 4]) # use the next day closing price for the dependent variable
    return np.array(dataX), np.array(dataY)

# Lag value definition
look_back = 2
# Splitting data in to train and test
trainX, trainY = create_lag_window_dataset(train, sentiment[0:train_size], look_back,use_sentiment= True)
testX, testY = create_lag_window_dataset(test, sentiment[train_size:len(scaled_prices)], look_back, use_sentiment= True)

## Set the seed
from numpy.random import seed
seed(1)
import tensorflow as tf
from tensorflow.python.framework.random_seed import set_random_seed
set_random_seed(2)
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Create default session
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

from tensorflow.python.keras import backend as K
from keras import __version__

if K.backend() == "tensorflow":
    import tensorflow as tf

    device_name = tf.test.gpu_device_name()
    if device_name == '':
        device_name = "None"
    print('Using TensorFlow version:', tf.__version__, ', GPU:', device_name)


print('Using Keras version:', __version__, 'backend:', K.backend())


K.set_session(sess)


# importing libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt
from sklearn.metrics import mean_squared_error

# Building model
start_time = time.time()

model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(trainX, trainY, epochs=100, batch_size=100, shuffle=False)
end_time = time.time()
execution_time = end_time-start_time
print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(execution_time)))

# Testing accuracy of the model
yhat_test = model.predict(testX)

rmse = sqrt(mean_squared_error(testY, yhat_test))
print(f'Test RMSE: {rmse}')


# Plotting ground truth vs Predicted

yhat_train = model.predict(trainX)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(trainY, label='Groundtruth', color='orange')
plt.plot(yhat_train, label='Predicted', color='purple')
plt.title("Training")
plt.ylabel("Scaled Price")
plt.legend(loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(testY, label='Groundtruth', color='orange')
plt.plot(yhat_test, label='Predicted', color='purple')
plt.title("Test")
plt.ylabel("Scaled Price")
plt.legend(loc='upper left')

plt.show()

# Processing the Predictions
a = np.zeros((yhat_test.shape[0], 4))
# Stacking the predictions for mapping
yhat_test = np.hstack([a, yhat_test])
# Inverse predictions transformations
yhat_test_inverse = scaler.inverse_transform(yhat_test)

# 7 - Plot price (Inverse transform)
#yhat_test_inverse = scaler.inverse_transform(yhat_test)
predicted_price = yhat_test_inverse[:, 4]

testY = testY.reshape(-1, 1)
testY = np.hstack([a, testY])
testY_inverse = scaler.inverse_transform(testY)
real_price = testY_inverse[:, 4]

plt.plot(real_price, label='Actual', color='royalblue')
plt.plot(predicted_price, label='Predicted', color='indianred')
plt.title("Predicted vs Actual")
plt.ylabel("Price")
plt.legend(loc='upper left')

plt.show()


