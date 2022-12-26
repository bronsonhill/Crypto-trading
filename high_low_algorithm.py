from data_processing import sma_data, import_data
import pandas as pd
import mplfinance as mpf


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


def find_price_extrema(df, extrema):
    '''accepts ohlc df and a list of tuples of indicator extrema and 
    their index and uses it to detect trends/price-movements and their 
    extrema. Returns a dataframe with columns datetime and Price'''
    extrema_list = []
    lower_index = extrema[0][0]
    # establishes initial trend
    if extrema[0][1] >= extrema[1][1]:
        up_trend = False
    else:
        up_trend = True

    extrema_list += [(None, None)] * (lower_index)
    # records price extrema with the guide of ma extrema
    prev_extreme = extrema[0][1]
    extrema = extrema[1:]
    for extreme in extrema:
        upper_index = extreme[0]
        trend_data = df.iloc[lower_index:upper_index]

        if up_trend:
            trend_str = 'up'
            extrema_str = 'max'
        else:
            trend_str = 'down'
            extrema_str = 'min'
        # print(f'Searching {trend_str} trend from index {lower_index} to {upper_index} for {extrema_str}...')

        # records max price point of up trend
        if up_trend:
            additional_extrema = len(trend_data)*[(None, None)]
            extreme_price = max(trend_data['High'])
            index = list(trend_data['High']).index(extreme_price)
            date = trend_data.index[index]
            additional_extrema[index] = (date, extreme_price)
            extrema_list += additional_extrema
            
            # print(f'...max is: {extreme_price}')
            # reverses trend for next trend
            up_trend = False
        
        # records min price point of down trend
        else:
            additional_extrema = len(trend_data)*[(None, None)]
            extreme_price = min(trend_data['Low'])
            index = list(trend_data['Low']).index(extreme_price)
            date = trend_data.index[index]
            additional_extrema[index] = (date, extreme_price)
            extrema_list += additional_extrema

            # print(f'...min is: {extreme_price}')
            # reverses trend for next trend
            if prev_extreme == extreme[0]:
                print('something fishy')
            up_trend = True
        lower_index = upper_index
        prev_extreme = extreme

        # # view chart of each trend movement 
        signal_data = pd.Series([y for (x, y) in additional_extrema])
        signal_plot = mpf.make_addplot(signal_data, type='scatter', markersize=100)
        trend_data.index = pd.to_datetime(trend_data.index, utc=True)
        mpf.plot(trend_data, title=f'{trend_str} trend', ylabel='Price', 
            type='candle', style='mike', addplot=signal_plot,
            volume=False)
    remaining_len = len(df.iloc[upper_index:])
    extrema_list += [(None, None)] * (remaining_len)
    extrema_df = pd.DataFrame(extrema_list, columns=['Datetime', 
            'Extreme Price'])
    return extrema_df


def high_low(ticker, interval, ma_length):
    '''using various functions returns a dataframe with datetime and
    extreme (high/low) price'''

    ohlc_df = import_data(ticker, interval, 800)
    ma_data = sma_data(ohlc_df, ma_length)
    ma_extrema_df = find_local_extrema(ma_data)

    return find_price_extrema(ohlc_df, ma_extrema_df)


def analyse_high_low(df):
    '''takes an extreme price dataframe and returns statistics'''
    i = 0

    for row in df.itertuples():
        if row[0] == None:
            df.drop(index=i)

    average_len = len(df)
    return

high_low('ES=F', '15m', 4)