from data_processing import sma_data, import_data
import pandas as pd


def find_local_extrema(cont_data, ma_length=0):
    '''detects local extrema in continous data, returning a list of 
    tuples with index of extrema and extrema value'''
    
    extrema = [(ma_length-1, cont_data[ma_length])]
    for i in range(1, len(cont_data) - 1):
        if cont_data[i] > cont_data[i-1] and cont_data[i] > cont_data[i+1]:
            # local maximum found
            extrema.append((i, cont_data[i]))
        elif cont_data[i] < cont_data[i-1] and cont_data[i] < cont_data[i+1]:
            # local minimum found
            extrema.append((i, cont_data[i]))
    return extrema


def find_price_extrema(df, extrema):
    '''accepts ohlc df and list of tuples of extrema and their index and
    returns a list of tuples with index of extrema and price extrema'''

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
        price_list = []
        upper_index_bracket = extreme[0]
        trend_data = df.iloc[lower_index_bracket:upper_index_bracket]
        # records max price point if in up trend
        if up_trend:
            for row in trend_data.itertuples():
                price_list.append((row[0], row[2]))
            price_extrema_data.append(max(price_list, key=lambda x: x[1]))
            up_trend = False
        # records min price point if in down trend
        else:
            for row in trend_data.itertuples():
                price_list.append((row[0], row[2]))
            price_extrema_data.append(min(price_list, key=lambda x: x[1]))
            up_trend = True
        lower_index_bracket = upper_index_bracket

    df = pd.DataFrame(price_extrema_data, columns=['Datetime', 'Extreme Price'])
    return df


def high_low(ticker, interval, ma_length):
    '''using various functions returns a dataframe with datetime and
    extreme (high/low) price'''

    ohlc_df = import_data(ticker, interval)
    ma_data = sma_data(ohlc_df, ma_length)
    ma_extrema_df = find_local_extrema(ma_data, ma_length)

    return find_price_extrema(ohlc_df, ma_extrema_df)


# print(high_low('GC=F', '15m', 7))