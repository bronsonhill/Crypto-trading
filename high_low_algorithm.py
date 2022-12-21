from data_processing import sma_data, averaged_price, import_data

import pandas as pd

df = import_data('Price_data/EURUSD=X/15m.csv')
ma_data = sma_data(averaged_price(df), 7)

def find_local_extrema(data, length=0):
    '''detects local extrema in data, returning a list of tuples with
    index of extrema and extrema value'''
    
    extrema = [(length-1, data[length])]
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            # local maximum found
            extrema.append((i, data[i]))
        elif data[i] < data[i-1] and data[i] < data[i+1]:
            # local minimum found
            extrema.append((i, data[i]))
    return extrema


def price_extrema(df, extrema):
    '''accepts ohlc df and list of extrema and returns a list of tuples
    with index of extrema and price extrema'''

    price_extrema_data = []
    
    # establishes initial trend
    extrema = iter(extrema)
    if next(extrema)[1] >= next(extrema)[1]:
        up_trend = False
    else:
        up_trend = True

    lower_index_bracket = 0
    # records price extrema according to ma extrema
    for extreme in extrema:
        upper_index_bracket = extreme[0]
        trend_data = df.iloc[lower_index_bracket:upper_index_bracket]
        # records max price point if in up trend
        if up_trend:
            for row in trend_data.itertuples():
                price_extrema_data.append((row[0], row[2]))
            up_trend = False
        # records min price point if in down trend
        else:
            for row in trend_data.itertuples():
                price_extrema_data.append((row[0], row[3]))
            up_trend = True
        lower_index_bracket = upper_index_bracket

    df = pd.DataFrame(price_extrema_data, columns=['Datetime', 'Extreme Price'])
    return df

print(price_extrema(import_data('Price_data/EURUSD=X/15m.csv'), find_local_extrema(ma_data, 7)))