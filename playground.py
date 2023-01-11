import pandas as pd
import numpy as np
import datetime
import json
     
with open('Info/yf_ticker_names.json', 'r') as fp:
     data = json.load(fp)

log = dict()
for ticker in data:
     log[ticker] = "2023-01-11"

with open('Info/data_retrieval_log.json', 'w') as fp:
     json.dump(log, fp)





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