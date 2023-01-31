import pandas as pd
import numpy as np
import datetime
import json
from binance import Client
import os
     
print(10000//10000)

client = Client(
            api_key='FuaLBWg3iJCPTTQrU1yewim305sUVvZwOzO4Xau75JLHP0lTlpK9V7bdbPSBZOwF', 
            api_secret='b8mt4w5PKu0gIvDoWHWSpGxdj8dWzesQ4RFCufcHf7l1D6LFMwQN2BAzyqN3lcTI'
    )

# timestamp = client._get_earliest_valid_timestamp('BTCUSDT', '1d')
# print(datetime.datetime.fromtimestamp(timestamp/1000))

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