import pandas as pd
import matplotlib.pyplot as plt
# https://github.com/matplotlib/mplfinance/blob/master/examples/addplot.ipynb
import mplfinance as mpf
import data_processing


def chart(ticker, interval, sma):
    ''''''
    filename = f'Price_data/{ticker}/{interval}.csv'
    df = data_processing.import_data(filename)
    df.index = pd.to_datetime(df.index, utc=True)
    
    # creatre sma plot
    if sma:
        sma_data = data_processing.sma_data(df, length=sma)
        sma_plot = mpf.make_addplot(sma_data)

    # prepares df for candle plot
    for name in df.columns:
        if name not in ('Open', 'High', 'Low', 'Close'):
            df.drop([name], axis=1, inplace=True)

    # Create the candle plot
    mpf.plot(df, type='candle', style='yahoo', addplot=sma_plot)


chart('EURUSD=X', '15m', 7)


