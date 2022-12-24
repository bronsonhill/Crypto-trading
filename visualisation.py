import pandas as pd
import matplotlib.pyplot as plt
# https://github.com/matplotlib/mplfinance/blob/master/examples/addplot.ipynb
import mplfinance as mpf
import data_processing
from high_low_algorithm import high_low


def chart_from_ticker(ticker, interval, signals=False, smas=[]):
    ''''''
    filename = f'Price_data/{ticker}/{interval}.csv'
    df = data_processing.import_data(ticker, interval, 1000)
    df.index = pd.to_datetime(df.index, utc=True)
    addplot = []
    # create sma plot
    for sma in smas:
        sma_data = data_processing.sma_data(df, length=sma)
        sma_plot = mpf.make_addplot(sma_data)
        addplot.append(sma_plot)
    
    # create signals plot
    if signals:
        signal_data = high_low(ticker, interval, smas[0])['Extreme Price']
        signal_plot = mpf.make_addplot(signal_data, type='scatter', markersize=60)
        addplot.append(signal_plot)

    # prepares df for candle plot
    for name in df.columns:
        if name not in ('Open', 'High', 'Low', 'Close'):
            df.drop([name], axis=1, inplace=True)

    # Create the candle plot
    mpf.plot(df, type='candle', style='yahoo', addplot=addplot)



chart_from_ticker('ZC=F', '15m', signals=True, smas=[4, 50])
