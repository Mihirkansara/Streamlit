import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import sys
import base64
sys.setrecursionlimit(2000) # Increase the recursion limit to 2000

# with open("stock.jpg", "rb") as img_file:
#     my_img = img_file.read()
# st.markdown(
#     f'<img src="data:stock/jpeg;base64,{base64.b64encode(my_img).decode()}" alt="background image" style="position: fixed; bottom: 0px; right: 0px; z-index: -1; opacity: 0.5; width: 100%; height: 100%;">',
#     unsafe_allow_html=True,
# )


page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://wallpapers-hub.art/wallpaper-images/340609.jpg");
  background-size: cover;
}
[data-testid="stSidebarContent"]{
  background-image: url("https://wallpapers-hub.art/wallpaper-images/340609.jpg");
  background-size: cover;
}
[data-testid="stHeader"]{
  background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

# start ='2010-01-01' 
# End = '2019-12-31'

st.title('Stock Price Prediction')
st.sidebar.header('Enter Stock Ticker Here')
Ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')
# use_input = st.text_input('Enter stock Ticker', 'AAPL')##############   
# df = data.DataReader(use_input,'yahoo')
data = yf.download(Ticker,start=start_date,end=end_date)
data
st.subheader('Data From Which You Select The Dates')
#Data Visualization
st.subheader('Closing price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(data.Close)
st.pyplot(fig)

#Data Visualization For Moving Average of 100 Days
st.subheader('Closing price vs 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)


#Data Visualization For Moving Average of 200 Days
st.subheader('Closing price vs 100MA & 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(data.Close,'b')
st.pyplot(fig)

# Data Train & Test Split 

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#load ML Model
model = load_model('model_LSTM.h5')

#Testing Part
past_100_days = data_training.tail(100)
#print(data_training.tail(100))
#print(isinstance(past_100_days, pd.DataFrame))
# final_df = past_100_days.append(data_testing, ignore_index=True)
final_df = pd.concat([past_100_days, data_testing])
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])
x_test,y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

st.subheader('Predicted vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'g',label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

df = pd.DataFrame(y_predicted)
print(df)

df.to_csv('predictions.csv', index=False)

import boto3
s3 = boto3.resource(
    service_name = 's3',
    region_name = 'us-east-2',
    aws_access_key_id = 'AKIAYKFZQFKQDGQD5U4Y',
    aws_secret_access_key='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'
)
#AWS Cloud
import os
import boto3
import logging
import s3fs
# AWS Credentials
def list_buckets():
        try:
                s3 = boto3.client('s3')
                response = s3.list_buckets()
                if response:
                        print('Bucket Exists.....')
                        for bucket in response['Buckets']:
                            print(f'{bucket["Name"]}')    
        except Exception as e:
                logging.error(e)
                return False
        return True
list_buckets()        

bucket = s3.Bucket('lstm-ml')
filename = 'predictions.csv'
key = 'test5.csv'
bucket.upload_file(Filename=filename, Key=key)

#For Download Predicted CSV
bucket.download_file(Key=key,Filename = 'finaldf.csv')
#download button for this csv file
st.download_button(
  label="Download Predicted CSV",
  data=open("finaldf.csv", "rb"),
  file_name="finaldf.csv",
  )