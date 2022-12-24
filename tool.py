import pandas as pd

intervals = ('1m', '2m', '5m', '15m', '30m', '1h')

for interval in intervals:
    dir = f'Price_data/ZC=F/{interval}.csv'
    df = pd.read_csv(dir)
    df = df.set_index('Datetime')
    df.to_csv(dir)
