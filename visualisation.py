import pandas as pd
import matplotlib.pyplot as plt
# https://github.com/matplotlib/mplfinance/blob/master/examples/addplot.ipynb
import mplfinance as mpf
import data_processing
from high_low_algorithm import find_price_extrema


def chart_from_ticker(ticker, interval, trend_signals=False, smas=[4]):
    ''''''
    filename = f'Price_data/{ticker}/{interval}.csv'
    df = data_processing.import_data(ticker, interval, 800)
    df.index = pd.to_datetime(df.index, utc=True)
    addplot = []
    # create sma plots
    for sma in smas:
        sma_data = data_processing.sma_data(df, length=sma)
        sma_plot = mpf.make_addplot(sma_data)
        addplot.append(sma_plot)
    
    # create trend signals plot using the first sma length
    if trend_signals:
        signal_data = find_price_extrema(ticker, interval, smas[0])['Extreme Price']
        print(type(signal_data))
        signal_plot = mpf.make_addplot(signal_data, type='scatter', markersize=60)
        addplot.append(signal_plot)

    # prepares df for candle plot
    for name in df.columns:
        if name not in ('Open', 'High', 'Low', 'Close', 'Volume'):
            df.drop([name], axis=1, inplace=True)

    # Create the candle plot
    mpf.plot(df, type='candle', style='binance', addplot=addplot)



chart_from_ticker('ES=F', '15m', trend_signals=True, smas=[4, 50])
