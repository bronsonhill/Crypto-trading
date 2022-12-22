import pandas as pd
import matplotlib.pyplot as plt
# https://github.com/matplotlib/mplfinance/blob/master/examples/addplot.ipynb
import mplfinance as mpf
import data_processing
from high_low_algorithm import high_low


def chart(ticker, interval, signals=False, sma=0):
    ''''''
    filename = f'Price_data/{ticker}/{interval}.csv'
    df = data_processing.import_data(ticker, interval)
    df.index = pd.to_datetime(df.index, utc=True)
    
    # create sma plot
    if sma:
        sma_data = data_processing.sma_data(df, length=sma)
        sma_plot = mpf.make_addplot(sma_data)
    
    # create signals plot
    if signals:
        signal_data = high_low(ticker, interval, sma)
        signal_plot = mpf.make_addplot(signal_data, type='scatter', markersize=100)
    

    # prepares df for candle plot
    for name in df.columns:
        if name not in ('Open', 'High', 'Low', 'Close'):
            df.drop([name], axis=1, inplace=True)

    # Create the candle plot
    mpf.plot(df, type='candle', style='yahoo', addplot=signal_plot)


chart('GC=F', '15m', signals=True, sma=7)


