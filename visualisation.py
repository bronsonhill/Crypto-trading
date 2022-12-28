import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# https://github.com/matplotlib/mplfinance/blob/master/examples/addplot.ipynb
import mplfinance as mpf
import data_processing
from high_low_algorithm import find_trend_extrema, analyse_trend_time, analyse_trend_movement


def chart_from_ticker(ticker, interval, trend_signals=False, smas=[4], data_points=300):
    ''''''
    filename = f'Price_data/{ticker}/{interval}.csv'
    df = data_processing.import_data(ticker, interval, data_points)
    df.index = pd.to_datetime(df.index, utc=True)
    addplot = []
    # create sma plots
    for sma in smas:
        sma_data = data_processing.sma_data(df, length=sma)
        sma_plot = mpf.make_addplot(sma_data)
        addplot.append(sma_plot)
    
    # create trend signals plot using the first sma length
    if trend_signals:
        signal_data = find_trend_extrema(ticker, interval, smas[0]
            , data_points=data_points)['Extreme Price']
        signal_plot = mpf.make_addplot(signal_data, type='scatter', markersize=60)
        addplot.append(signal_plot)

    # prepares df for candle plot
    for name in df.columns:
        if name not in ('Open', 'High', 'Low', 'Close', 'Volume'):
            df.drop([name], axis=1, inplace=True)

    # Create the candle plot with custom style
    mc = mpf.make_marketcolors(up='#1FFF56',down='#FF461F',inherit=True)
    style  = mpf.make_mpf_style(base_mpf_style='nightclouds',marketcolors=mc)
    mpf.plot(df, type='candle', title=f'\n\n{ticker} {interval} Chart', style=style, addplot=addplot)


def hist(series, title=''):
    ''''''
    # first remove outliers from data
    z = np.abs(stats.zscore(series))
    series = series.drop(series.index[list(np.where(z > 3))])

    # next create bin size
    maximum = max(list(series))
    minimum = min(list(series))
    binwidth = (maximum) / 35
    bins = np.arange(minimum, maximum + binwidth, binwidth)

    # now plot histogram
    plt.hist(series, color='grey', bins=bins)
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('Time Elapsed (Hours)')
    plt.show()
    return

series = analyse_trend_time('SI=F', '1m', 5)['Time Elapsed']
series = analyse_trend_movement('BTC-USD', '1m', 7)['Price Movement']
hist(series, 'Histogram\n')


# chart_from_ticker('ES=F', '1m', trend_signals=True, smas=[5, 50], data_points=3000)
