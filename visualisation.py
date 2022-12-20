import pandas as pd
import matplotlib.pyplot as plt
# https://github.com/matplotlib/mplfinance/blob/master/examples/addplot.ipynb
import mplfinance as mpf
import data_processing


def chart(ticker, interval):
    ''''''
    filename = f'Price_data/{ticker}/{interval}.csv'
    df = data_processing.import_data(filename)
    df.index = pd.to_datetime(df.index, utc=True)
    indicator = data_processing.add_sma(df, length=4)

    for name in df.columns:
        # prepares df for candle plot
        if name not in ('Open', 'High', 'Low', 'Close'):
            df.drop([name], axis=1, inplace=True)


    # Create the indicator plot
    indicator_plot = mpf.make_addplot(indicator)

    # Create the candle plot
    mpf.plot(df, type='candle', style='yahoo', addplot=indicator_plot)


chart('EURUSD=X', '15m')


