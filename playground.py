import pandas as pd
import numpy as np
import datetime

df = pd.read_csv('EURUSD_D1.csv')

new_col = []*df.shape[0]

df['Dividends'] = np.nan
df['Stock Splits'] = np.nan


df['Datetime'] = pd.to_datetime(df.Datetime)
df.to_csv('new.csv', index=False)
# format_str = "%Y-%m-%d %z"
# recent_date = datetime.datetime.strptime()

print(df)

