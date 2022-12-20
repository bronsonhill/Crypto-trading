import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpl 


def chart(filename):
    df = pd.read_csv(filename)
    df.index = pd.to_datetime(df.index)
    for name in df.columns:
        if name not in ('Open', 'High', 'Low', 'Close'):
            df.drop([name], axis=1, inplace=True)
    mpl.plot(df, type='candle', title=filename, style='yahoo')

chart('Price data/CL=F_1h.csv')


