from data_processing import sma_data, import_data
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt


def find_local_extrema(cont_data):
    '''detects local extrema in continous data, returning a list of 
    tuples with index of extrema and extrema value'''
    
    extrema = []
    for i in range(1, len(cont_data) - 1):
        if (cont_data[i] >= cont_data[i-1] and cont_data[i] > cont_data[i+1]) or \
            (cont_data[i] > cont_data[i-1] and cont_data[i] >= cont_data[i+1]):
            # local maximum found
            extrema.append((i, cont_data[i]))
        elif (cont_data[i] < cont_data[i-1] and cont_data[i] <= cont_data[i+1]) \
            or (cont_data[i] <= cont_data[i-1] and cont_data[i] < cont_data[i+1]):
            # local minimum found
            extrema.append((i, cont_data[i]))
    return extrema


def find_trend_extrema(ticker, interval, sma_length, data_points):
    '''For a ticker and interval returns a dataframe with the extreme
    prices of trends defined by a moving average of 'sma_length'
    Returns a dataframe with columns datetime and Price'''

    df = import_data(ticker, interval, data_points)
    sma_list = list(sma_data(df, sma_length))
    extrema = find_local_extrema(sma_list)
    extrema_list = []
    lower_i = extrema[0][0]

    # establishes initial trend
    if extrema[0][1] >= extrema[1][1]:
        up_trend = False
    else:
        up_trend = True
    # extrema_list becomes dataframe. Indexes are preserved to match ohlc data
    extrema_list += [(None, None)] * lower_i
    # records price extrema with the guide of ma extrema
    i = 1
    while i < (len(extrema) - 1):
        extreme = extrema[i][1]
        extreme_i = extrema[i][0]
        upper_i = extrema[i][0]
        trend_data = df.iloc[lower_i:upper_i]
        additional_extrema = len(trend_data)*[(None, None)]

        # records max price point of up trend
        if up_trend:
            extreme_price = max(trend_data['High'])
            index = list(trend_data['High']).index(extreme_price)
            date = trend_data.index[index]
            additional_extrema[index] = (date, extreme_price)
            extrema_list += additional_extrema
            
            # reverses trend for next trend (unless edge case: \--/
            if sma_list[extreme_i-1] == extreme and (sma_list[extreme_i-2] > extreme and \
                sma_list[extreme_i+1] > extreme):
                up_trend = True
            else:
                up_trend = False
        
        # records min price point of down trend
        else:
            extreme_price = min(trend_data['Low'])
            index = list(trend_data['Low']).index(extreme_price)
            date = trend_data.index[index]
            additional_extrema[index] = (date, extreme_price)
            extrema_list += additional_extrema

            # reverses trend for next trend (unless edge case: /--\
            if sma_list[extreme_i-1] == extreme and sma_list[extreme_i-2] < extreme and \
                sma_list[extreme_i+1] < extreme:
                up_trend = False
            else:
                up_trend = True
        lower_i = upper_i
        i += 1

        # # view chart of each trend movement 
        # signal_data = pd.Series([y for (x, y) in additional_extrema])
        # signal_plot = mpf.make_addplot(signal_data, type='scatter', markersize=100)
        # trend_data.index = pd.to_datetime(trend_data.index, utc=True)
        # mpf.plot(trend_data, title=f'{trend_str} trend', ylabel='Price', 
        #     type='candle', style='mike', addplot=signal_plot,
        #     volume=False)
    
    remaining_len = len(df.iloc[upper_i:])
    extrema_list += [(None, None)] * (remaining_len)

    return pd.DataFrame(extrema_list, columns=['Datetime', 'Extreme Price'])


def analyse_trend_time(ticker, interval, sma_length, data_points):
    '''takes a trend extrema dataframe and returns a dataframe with
    columns 'Datetime', of the start of the trend, and Time Elapsed'''
    time_elapsed_col = []
    iter_dates = []
    df = find_trend_extrema(ticker, interval, sma_length, data_points)
    # creates a list of datetimes to iterate through
    for datetime in list(df['Datetime']):
        if not pd.isnull(datetime):
            iter_dates.append(datetime)
    
    # creates a list of the time elapsed in each trend
    i = 1
    while i < len(iter_dates): 
        time_elapsed_col.append((iter_dates[i] - iter_dates[i-1]).total_seconds() / 3600)
        i += 1
    iter_dates.pop()
    # saves to dataframe
    time_elapsed_df = pd.DataFrame(
        {'Datetime': iter_dates,
         'Time Elapsed': time_elapsed_col
        })

    return time_elapsed_df

