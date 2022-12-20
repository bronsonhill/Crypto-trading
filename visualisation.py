import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpl 
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
    
    print(df)

    # Create a new figure and set the title
    fig = plt.figure()
    fig.suptitle(ticker + ' ' + interval)

    # Create the candle plot
    mpl.plot(df, type='candle', style='yahoo')

    # Create the indicator plot
    plt.plot(indicator)

    # Display the plots
    plt.show()


chart('EURUSD=X', '15m')


