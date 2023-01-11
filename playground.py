import pandas as pd
import numpy as np
import datetime
import json

YF_TICKER_NAMES = {'PL=F': 'Platinum', 'BTC-USD': 'Bitcoin USD', 
        'CL=F': 'Crude Oil', 
        'HG=F': 'Copper', 
        'ZW=F': 'Wheat', 
        'ES=F': 'E-Mini S&P 500', 
        'ETH-USD': 'Ethereum USD', 
        'EURUSD=X': 'EURUSD',
        'GC=F': 'Gold USD/oz',
        'PA=F': 'Palladium',
        'SI=F': 'Silver USD/oz',
        'TSLA': 'Tesla Shares',
        'ZO=F': 'Oat'
        }


with open('Info/yf_ticker_names.json', 'w') as fp:
     json.dump(YF_TICKER_NAMES, fp)
     
with open('Info/yf_ticker_names.json', 'r') as fp:
     data = json.load(fp)
     print(data)





# for importing ea trading data
'''
dir = 'Price_data/ADownloads/BRENTCMDUSD_M1.csv'
df = pd.read_csv(dir)
df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']

df['Datetime'] = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M")
     for x in df['Datetime']]

# df['Datetime'] = df['Datetime'].dt.tz_localize('GMT').dt.tz_convert('EST')
df['Datetime'] = df['Datetime'].dt.tz_localize('EST')

df.to_csv('Price_data/ADownloads/new.csv', index=False)'''