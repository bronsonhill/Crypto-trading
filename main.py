import datetime
import time
import yfinance
from binance import Client
import numpy as np
from scipy import stats
import pandas as pd
import json
import os
import pytz
import matplotlib.pyplot as plt
import mplfinance as mpf

start_time = time.time()
# the days of history available from yfinance
YF_HISTORY_LIMIT = {'1m': 30, '2m': 60, '5m': 60, '15m': 60, '30m': 60, 
'1h': 730, '1d': 730, '1wk': 10000, '1mo': 300}

# the days of history requestable from yfinance per request
YF_REQUEST_LIMIT = {'1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60, 
'1h': 730, '1d': 730, '1wk': 10000, '1mo': 10000}

# the minutes in each interval
INTERVAL_MINUTES = {'1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30, 
'1h': 60, '1d': 1140, '1wk': 10080, '1mo': 43829}

# the limits to trend time in hours - used to remove outliers
TREND_TIME_LIMITS = {'1m': 5, '2m': 10, '5m': 20, '15m': 24, '30m': 48, 
'1h': 96, '1d': 0, '1wk': 0, '1mo': 0}

TIMEZONE = {'=X': 'Etc/GMT', '=F': 'EST', '-': 'Etc/UTC', 
        'TSLA': 'EST', '^': 'EST'}


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


def bin_size(series, binwidth=0):
    maximum = max(list(series))
    minimum = min(list(series))
    if not binwidth:
        if minimum < 0:
            binwidth = (maximum) / 25
        else:
            binwidth = (maximum) / 50

    np.arange(minimum, maximum + binwidth, binwidth)
    return np.arange(minimum, maximum + binwidth, binwidth)


def remove_outliers(series, deviations=3):
    '''removes outliers using median absolute deviation threshold'''
    series = np.array(series)
    median = np.median(series)
    diff = np.sqrt(np.sum((series-median)**2, axis=-1))
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.675 * diff / med_abs_deviation
    
    return np.where(series > deviations)


def update_database(days=7):
    '''Retrieves all Ticker data that has not been updated in specified 
    number of days'''
    days = datetime.timedelta(days=days)
    with open('Info/data_retrieval_log.json', 'r') as fp:
        data = json.load(fp)

    for ticker, date in data.items():
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
        if (datetime.datetime.today() - date) >= days:
            Ticker(ticker).update_data()

    return


class Ticker:
    def __init__(self, ticker):
        self.ticker = ticker
        self.tf_1m = Pair(ticker, '1m')
        self.tf_2m = Pair(ticker, '2m')
        self.tf_5m = Pair(ticker, '5m')
        self.tf_15m = Pair(ticker, '15m')
        self.tf_30m = Pair(ticker, '30m')
        self.tf_1h = Pair(ticker, '1h')
        self.tf_1d = Pair(ticker, '1d')
        self.tf_1wk = Pair(ticker, '1wk')
        self.tf = Pair(ticker, '1mo')
    
    def update_data(self):
        self.tf_1m.retrieve_data()
        self.tf_2m.retrieve_data()
        self.tf_5m.retrieve_data()
        self.tf_15m.retrieve_data()
        self.tf_30m.retrieve_data()
        self.tf_1h.retrieve_data()
        self.tf_1d.retrieve_data()
        self.tf_1wk.retrieve_data()
        self.tf.retrieve_data()

        # stores date of data retrieval for future reference
        with open('Info/data_retrieval_log.json', 'r') as fp:
            data = json.load(fp)
            today = datetime.datetime.today().strftime("%Y-%m-%d")
            data[str(self.ticker)] = today
        with open('Info/data_retrieval_log.json', 'w') as fp:
            json.dump(data, fp)


class Pair:
    def __init__(self, ticker, interval):
        self.ticker = ticker
        self. interval = interval

        # intialises ticker name
        with open('Info/yf_ticker_names.json', 'r') as fp:
            ticker_names = json.load(fp)
            
            # in the case that it is recorded already
            if ticker in ticker_names:
                self.ticker__name = ticker_names[ticker]
            else:
                name = input(f'The ticker name for {ticker} has not yet been recorded. Enter the name you would like it to be recorded as: ')
                ticker_names[ticker] = name
        
        # in the case it is not yet recorded
        with open('Info/yf_ticker_names.json', 'w') as fp:
            json.dump(ticker_names, fp)

        self.dir = f'Price_data/{self.ticker}/{self.interval}.csv'
    

    def __str__(self):
        return self.ticker + ' ' + self.interval
    
    
    def yfinance_fetch_ohlc(self, start, end):
        '''fetches ohlc dataframe for ticker for interval and start and end time'''
        ticker = yfinance.Ticker(self.ticker)
        df = ticker.history(start=start, end=end, interval=self.interval)
        return df

    
    def binance_fetch_ohlc(self, start):
        client = Client(
            api_key='FuaLBWg3iJCPTTQrU1yewim305sUVvZwOzO4Xau75JLHP0lTlpK9V7bdbPSBZOwF', 
            api_secret='b8mt4w5PKu0gIvDoWHWSpGxdj8dWzesQ4RFCufcHf7l1D6LFMwQN2BAzyqN3lcTI'
    )
        historical_data = client.get_historical_klines(self.ticker, 
            self.interval, start)
        
        df = pd.DataFrame(historical_data, columns=['Datetime', 'Open', 
        'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 
        'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore'])

        df['Datetime'] = pd.to_datetime(df['Open Time']/1000, unit='s')
        df['Close Time'] = pd.to_datetime(df['Close Time']/1000, unit='s')

        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'Quote Asset Volume', 'TB Base Volume', 'TB Quote Volume']
        
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)

        return df
    

    def file_exists(self):
        '''tests whether price data file already exists and returns last 
        price date if so, otherwise returns false'''
        dir = self.dir
        # tests file exists
        if os.path.exists(dir):
            df = pd.read_csv(dir)

            # determines the required datetime format
            if self.interval in ['1d', '1wk', '1mo']:
                format_str = "%Y-%m-%d"
                recent_date = datetime.datetime.strptime(
                df.iloc[-2,0:1].values[0],format_str) + datetime.timedelta(hours=12)
                
                return recent_date

            format_str = "%Y-%m-%d %H:%M:%S%z"
            recent_date = datetime.datetime.strptime(
                df.iloc[-2,0:1].values[0],format_str)

            # gives the time of most recent price data

            return recent_date
        
        else:
            return False


    def import_from_database(self, data_points=0):
        '''collects specified price data from database and returns it in
        pandas dataframe'''
        
        dir = self.dir
        # imports csv to df
        df = pd.read_csv(dir, index_col=0, parse_dates=True)
        df.drop(df.columns[[4, 5, 6]], axis=1, inplace=True)
        # shortens dataframe if requested
        if data_points:
            df = df.iloc[len(df)-data_points:len(df)]
        return df
    

    def averaged_price(self, data_points=0):
        '''returns df with column of averaged open, high, low and close
        called "price"'''
        df = self.import_from_database(data_points)
        average_price = []
        # averages ohlc for each row and adds it to list
        rows = df.itertuples()
        for row in rows:
            sum_price = float(row[1]) + float(row[2]) + float(row[3]) + float(row[4])
            average_price.append(sum_price/4)
        
        # adds column with list
        df['Price'] = average_price
        return df


    def sma_data(self, length, data_points=0):
        '''from price df adds a column of SMA with specified length'''
        df = self.import_from_database(data_points)
        # determines if Price column must be added
        if 'Price' in df.columns:
            None
        else:
            df = self.averaged_price(data_points)

        # returns moving average
        return df['Price'].rolling(length).mean()

    
    def ema_data(self, length, data_points=0):
        '''from price df returns a lsit of EMA values of specified 
        length. Formula: \n
        EMA = K x (Current Price - Previous EMA) + Previous EMA,
        where k = 2/(n+1)'''
        ema_data = []
        k = 2/(length+1)
        df = self.import_from_database(data_points)
        
        # determines if Price column must be added
        if 'Price' in df.columns:
            None
        else:
            df = self.averaged_price(data_points)

        price_data = list(df['Price'])
        initial_value = 0

        # calculates first value of ema with sma formula
        for i in range(length):
            initial_value += price_data[i]
            ema_data.append(float('nan'))
        ema_data[-1]= initial_value / length

        # calculates rest of ema values
        while i < len(price_data)-1:
            ema_data.append(k * (price_data[i]-ema_data[i]) + ema_data[i])
            
            i += 1
        return ema_data


    def clean_data(self, series, zscore = True, interval_limit=False, sds=3):
        ''''''
        # when requested, will drop all trends which are over limit
        if interval_limit:
            if TREND_TIME_LIMITS[self.interval]:
                indexes = []
                i = 0
                for row in series:
                    if row >= TREND_TIME_LIMITS[self.interval]:
                        indexes.append(i)
                
                series = series.drop(indexes)

        # drops value from series where zscore is greater than the 
        # specified sds
        if zscore:
            z = np.abs(stats.zscore(series))
            series = series.drop(series.index[list(np.where(z > sds))])
        
        return series


    def retrieve_data(self):
        '''adds requested data to price data folder as csv'''
        ticker = self.ticker
        interval = self.interval
        dir = self.dir

        # creates ticker folder if required
        if not os.path.exists(f'Price_data/{ticker}'):
            os.makedirs(f'Price_data/{ticker}')

        # sets start date of request according to whether file exists or not
        exist_bool = self.file_exists()
        if exist_bool:
            start = exist_bool
        
        # sets start date to oldest date allowed by yfinance if some data not
        # already pre-existing in database
        else:
            try:
                start = datetime.datetime.now() - datetime.timedelta(
                    days=YF_HISTORY_LIMIT[interval]) + datetime.timedelta(hours=16)

            except:
                print(f'Try interval: {[interval for interval in YF_HISTORY_LIMIT.keys()]}')
                return
        
        df = pd.DataFrame()
        # makes requests for data from api and appends to pandas dataframe 
        if interval != '1d' and interval != '1wk' and interval != '1mo':
            for ticker_identifier, time_zone in TIMEZONE.items():
                if ticker_identifier in ticker:
                    break
            today = datetime.datetime.now(pytz.timezone('Australia/Victoria'))
            start = start.astimezone(pytz.timezone('Australia/Victoria'))
        else:
            today = datetime.datetime.now()

        end = start

        while end < today:
            end = start + datetime.timedelta(days=YF_REQUEST_LIMIT[interval])
            
            if not df.empty:
                df = pd.concat([df, self.yfinance_fetch_ohlc(start, end)])
            else:
                df = pd.DataFrame(self.yfinance_fetch_ohlc(start, end))
            start = end
        
        # either appends dataframe to existing csv or creates new one
        if exist_bool:
            # deletes last row of csv file
            csv_df = pd.read_csv(dir)
            csv_df.drop(csv_df.tail(2).index, inplace=True)
            csv_df.to_csv(dir, index=False, index_label='Datetime')
            df.to_csv(dir, mode='a', index=True, header=False)
        else:
            df.to_csv(dir)
        
        
        return


    def find_trend_extrema(self, sma_length, data_points=0):
        '''For a ticker and interval returns a dataframe with the extreme
        prices of trends defined by a moving average of 'sma_length'
        Returns a dataframe with columns datetime and Price'''

        df = self.import_from_database(data_points)
        sma_list = list(self.sma_data(sma_length, data_points))
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


    def analyse_trend_time(self, sma_length, data_points=0):
        '''takes a trend extrema dataframe and returns a dataframe with
        columns 'Datetime', of the start of the trend, and Time Elapsed'''
        date_col = []
        time_elapsed_col = []
        iter_dates = []
        df = self.find_trend_extrema(sma_length, data_points)

        # creates a list of datetimes to iterate through
        for datetime in list(df['Datetime']):
            if not pd.isnull(datetime):
                iter_dates.append(datetime)
        
        # creates a list of the time elapsed in each trend
        i = 1
        while i < len(iter_dates): 
            interval = (iter_dates[i] - iter_dates[i-1]).total_seconds() / 3600
            if interval <= TREND_TIME_LIMITS[self.interval] or \
                    TREND_TIME_LIMITS[self.interval] == 0:
                time_elapsed_col.append(interval)
                date_col.append(iter_dates[i-1])
            i += 1
        iter_dates.pop()
        # saves to dataframe
        time_elapsed_df = pd.DataFrame(
            {'Datetime': date_col,
            'Time Elapsed': time_elapsed_col
            })

        return time_elapsed_df


    def analyse_trend_movement(self, sma_length, data_points=0):
        '''pickle'''
        date_col = []
        price_movement_col = []
        iter_dates = []
        iter_prices = []
        df = self.find_trend_extrema(sma_length, data_points)

        # creates a list of datetimes and prices to iterate through
        for datetime, price in df.itertuples(index=False):
            if not pd.isnull(datetime):
                iter_dates.append(datetime)
                iter_prices.append(price)
        # creates a list of the time elapsed in each trend
        i = 1
        while i < len(iter_dates):
            interval = (iter_dates[i] - iter_dates[i-1]).total_seconds() / 3600
            if interval <= TREND_TIME_LIMITS[self.interval] or \
                    TREND_TIME_LIMITS[self.interval] == 0:
                percent_move = (iter_prices[i] - iter_prices[i-1])/iter_prices[i-1]*100
                price_movement_col.append(percent_move)
                date_col.append(iter_dates[i-1])
            i += 1
        iter_dates.pop()
        # saves to dataframe
        price_change_df = pd.DataFrame(
            {'Datetime': date_col,
            'Price Movement': price_movement_col
            })

        return price_change_df
    

    def ohlc_chart(self, trend_signals=True, smas=[4], emas=[], data_points=300):
        ''''''
        ticker = self.ticker
        interval = self.interval
        df = self.import_from_database(data_points)
        df.index = pd.to_datetime(df.index, utc=True)
        addplot = []
        # create sma plots
        for sma in smas:
            sma_data = self.sma_data(length=sma, data_points=data_points)
            sma_plot = mpf.make_addplot(sma_data)
            addplot.append(sma_plot)
        
        for ema in emas:
            sma_data = self.ema_data(length=sma, data_points=data_points)
            sma_plot = mpf.make_addplot(sma_data)
            addplot.append(sma_plot)
        
        # create trend signals plot using the first sma length
        if trend_signals:
            signal_data = self.find_trend_extrema(smas[0]
                , data_points=data_points)['Extreme Price']
            signal_plot = mpf.make_addplot(signal_data, type='scatter', markersize=60)
            addplot.append(signal_plot)

        # prepares df for candle plot
        for name in df.columns:
            if name not in ('Open', 'High', 'Low', 'Close'):
                df.drop([name], axis=1, inplace=True)

        # Create the candle plot with custom style
        mc = mpf.make_marketcolors(up='#1FFF56',down='#FF461F',inherit=True)
        style  = mpf.make_mpf_style(base_mpf_style='nightclouds',marketcolors=mc)
        mpf.plot(df, type='candle', title=f'\n\n{ticker} {interval} Chart', style=style, addplot=addplot, show_nontrading=False)
        return


    def hist_chart(self, sma_length=4):
        '''returns a figure of histograms where the integer inputted is
        the length of the sma used to define the trends'''
        fig, axs =  plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
        fig.suptitle(f'{self.ticker}{self.interval}', fontsize=14)
        trend_movement_data = self.clean_data(
                self.analyse_trend_movement(sma_length)['Price Movement'], 
                interval_limit=True
            )
        trend_time_data = self.clean_data(
                self.analyse_trend_time(sma_length)['Time Elapsed'], 
                interval_limit=True
            )
        
        axs[0][0].hist(trend_movement_data, bins=bin_size(trend_movement_data))
        axs[0][0].set(xlabel='Price Movement (%)', ylabel='Frequency', title='Trend Price Movement Histogram')

        axs[0][1].hist(trend_time_data, bins=bin_size(trend_time_data, INTERVAL_MINUTES[self.interval]/60))
        axs[0][1].set(xlabel='Elasped Time (Hrs)', ylabel='Frequency', title='Trend Elapsed Time Histogram')
        
        axs[1][0].plot(trend_movement_data)
        axs[1][0].set(xlabel='Trend No.', ylabel='Trend Movement (%)', title='\n\nTrend Elapsed Time Histogram')
        axs[1][1].plot(trend_time_data)
        axs[1][1].set(xlabel='Trend No.', ylabel='Elapsed time', title='\n\nTrend Elapsed Time')
        
        return
    

    def scatter_chart(self, sma_length=4):
        ''''''
        fig, axs =  plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
        fig.suptitle(f'{self.ticker}{self.interval}', fontsize=14)
        trend_movement_data = list(self.analyse_trend_movement(sma_length)['Price Movement'])
        trend_time_data = list(self.analyse_trend_time(sma_length)['Time Elapsed'])
        # print(trend_movement_data, trend_time_data)
        # x is the % of a down trend following an up trend
        x = []
        x1 = []
        x2 = []
        # y is the % of the up trend
        y = []
        y1 = []
        y2 = []
        i = 0
        while i < len(trend_movement_data)-1:
            row = trend_movement_data[i]
            next_row = trend_movement_data[i+1]
            if row > 0:
                if next_row < 0:
                    x.append(row)
                    y1.append(trend_time_data[i])
                    y.append(next_row)
                    x1.append(trend_time_data[i+1])
            # Collecting data of opposite
            else:
                if next_row > 0:
                    x2.append(row)
                    y2.append(next_row)
            i += 1


        axs[0][0].scatter(x, y, linewidths=0.5)
        axs[0][0].set(xlabel='Uptrend move (%)', 
            ylabel='Following downtrend move (%)', 
            title='Consecutive trend sizes (Up -> Down)'
            )
        axs[0][1].scatter(x2, y2, linewidths=0.5)
        axs[0][1].set(xlabel='Downtrend move %', 
            ylabel='Following uptrend move (%)', 
            title='Consecutive trend sizes (Down -> Up)'
            )
        axs[1][0].scatter(x1, y1, linewidths=0.5)
        axs[1][0].set(xlabel='Following downtrend time (Hrs)', 
            ylabel='Uptrend time (Hrs)', 
            title='Uptrend time and the following downtrend time'
            )
        axs[1][1].scatter(y1, y, linewidths=0.5)
        axs[1][1].set(xlabel='Uptrend time (Hrs)', 
            ylabel='Following downtrend move (%)', 
            title='Uptrend time and the following downtrend time'
            )
        
        print(f'Correlation of uptrend and following downtrend: {np.corrcoef(x, y)[0][1]}')
        print(f'Correlation of downtrend and following uptrend: {np.corrcoef(x2, y2)[0][1]}')
        print(f'Correlation of uptrend time and following downtrend time: {np.corrcoef(x1, y1)[0][1]}')
        print(f'Correlation of uptrend time and following downtrend move: {np.corrcoef(y1, y)[0][1]}')

        return

    




print(f'------ Total Runtime: {(time.time()-start_time):.3f} seconds ------')
