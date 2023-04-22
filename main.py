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
import csv

start_time = time.time()
# the days of history available from yfinance
YF_HISTORY_LIMIT = {'1m': 30, '2m': 60, '5m': 60, '15m': 60, '30m': 60, 
'1h': 730, '1d': 730, '1wk': 10000, '1mo': 300}

# the days of history requestable from yfinance per request
YF_REQUEST_LIMIT = {'1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60, 
'1h': 730, '1d': 730, '1wk': 10000, '1mo': 10000}

# the minutes in each interval
YF_INTERVAL_MINUTES = {'1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30, 
'1h': 60, '1d': 1140, '1wk': 10080, '1mo': 43829}

BINANCE_INTERVAL_MINUTES = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
 '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720, '1d': 1140,
 '3d': 3420, '1w': 10080, '1M': 43829}

BINANCE_HISTORY_LIMIT = {'1m': 2630000000, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
 '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720, '1d': 1140,
 '3d': 3420, '1w': 10080, '1M': 43829}

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
    # days required since last update
    days = datetime.timedelta(days=days)
    
    # binance data
    with open('Info/log_binance.json', 'r') as fp:
        data = json.load(fp)

    for ticker, lst in data.items():
        # exception in the case that an update date has not been logged
        try: 
            date = datetime.datetime.strptime(lst[1], "%Y-%m-%d")
            if (datetime.datetime.today() - date) >= days:
                pair = Pair(ticker, 'binance')
                pair.update_data()
        except: 
            Pair(ticker, 'binance').update_data()

    # yahoo data
    with open('Info/log_yahoo.json', 'r') as fp:
        data = json.load(fp)

    for ticker, lst in data.items():
        # exception in the case that an update date has not been logged
        try: 
            date = datetime.datetime.strptime(lst[1], "%Y-%m-%d")
            if (datetime.datetime.today() - date) >= days:
                Pair(ticker, 'yahoo').update_data()
        except: 
            Pair(ticker, 'yahoo').update_data()

    return


class Pair:
    def __init__(self, ticker, source):
        self.ticker = ticker
        self.source = source
        if source == 'yahoo':
            self.tf_1m = Timeframe(ticker, '1m', source)
            self.tf_2m = Timeframe(ticker, '2m', source)
            self.tf_5m = Timeframe(ticker, '5m', source)
            self.tf_15m = Timeframe(ticker, '15m', source)
            self.tf_30m = Timeframe(ticker, '30m', source)
            self.tf_1h = Timeframe(ticker, '1h', source)
            self.tf_1d = Timeframe(ticker, '1d', source)
            self.tf_1wk = Timeframe(ticker, '1wk', source)
            self.tf_1mo = Timeframe(ticker, '1mo', source)
            self.pairs = [self.tf_1m, self.tf_2m, self.tf_5m, self.tf_15m,
                self.tf_30m, self.tf_1h, self.tf_1d, self.tf_1wk, self.tf_1mo]
        
        elif source == 'binance':
            self.tf_1m = Timeframe(ticker, '1m', source)
            self.tf_5m = Timeframe(ticker, '5m', source)
            self.tf_15m = Timeframe(ticker, '15m', source)
            self.tf_30m = Timeframe(ticker, '30m', source)
            self.tf_1h = Timeframe(ticker, '1h', source)
            self.tf_4h = Timeframe(ticker, '4h', source)
            self.tf_1d = Timeframe(ticker, '1d', source)
            self.tf_1w = Timeframe(ticker, '1w', source)
            self.pairs = [self.tf_1m, self.tf_5m, self.tf_15m,
                self.tf_30m, self.tf_1h, self.tf_4h, self.tf_1d, self.tf_1w]
            return
    

    def update_data(self):
        ''''''
        if self.source == 'yahoo':
            for pair in self.pairs:
                pair.yahoo_fetch_ohlc()
            log_name = 'yahoo'
        elif self.source == 'binance':
            for pair in self.pairs:
                pair.binance_fetch_ohlc()
            log_name = 'binance'

        # stores date of data retrieval for future reference
        with open(f'Info/log_{log_name}.json', 'r') as fp:
            data = json.load(fp)
            today = datetime.datetime.today().strftime("%Y-%m-%d")
            key_pair = data[str(self.ticker)]
            key_pair[1] = today
            data[str(self.ticker)] = key_pair
        with open(f'Info/log_{log_name}.json', 'w') as fp:
            json.dump(data, fp, indent=4)


class Timeframe:
    def __init__(self, ticker: str, interval: str, source: str
    , start_date: datetime=None, end_date: datetime=None):
        self.ticker = ticker
        self.interval = interval
        self.source = source
        self.start_date = start_date
        self.end_date = end_date

        self.dir = f'../Price_data/{source}/{self.ticker}/{self.interval}.csv'

        # intialises ticker name
        with open(f'Info/log_{source}.json', 'r') as fp:
            ticker_names = json.load(fp)
            # in the case that it is recorded already
            if ticker in ticker_names:
                self.ticker__name = ticker_names[ticker]
            # in the case it needs to be specified by user
            else:
                name = input(f'The ticker name for {ticker} has not yet been recorded. Enter the name you would like it to be recorded as: ')
                # naxme = ''
                ticker_names[ticker] = name
        
        # in the case it is not yet recorded
        with open(f'Info/ticker_names_{source}.json', 'w') as fp:
            json.dump(ticker_names, fp, indent=4)
    

    def __str__(self):
        string = f'Ticker: {self.ticker}\nInterval: {self.interval}\n\
Source: {self.source}'
        return string
    
    
    def yfinance_fetch_ohlc(self, start, end):
        '''fetches ohlc dataframe for ticker for interval and start and end time'''
        ticker = yfinance.Ticker(self.ticker)
        df = ticker.history(start=start, end=end, interval=self.interval)
        return df


    def binance_fetch_ohlc(self):
        '''fetches data from binance api'''
        # create client instance
        client = Client(
            api_key='FuaLBWg3iJCPTTQrU1yewim305sUVvZwOzO4Xau75JLHP0lTlpK9V7bdbPSBZOwF', 
            api_secret='b8mt4w5PKu0gIvDoWHWSpGxdj8dWzesQ4RFCufcHf7l1D6LFMwQN2BAzyqN3lcTI'
    )
        # determines start date according to whether data already exists in 
        # database
        file_exists = self.file_exists()
        if file_exists:
            # should not have to read csv
            df = pd.read_csv(self.dir)
            df.drop(df.tail(1).index, inplace=True)
            timestamp = file_exists
        else:
            timestamp = client._get_earliest_valid_timestamp(self.ticker, self.interval)
            df = pd.DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 
            'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 
            'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore'])
            print(f'Retrieving data since {datetime.datetime.fromtimestamp(timestamp/1000)}')
        print(f'Requesting data for {self.ticker} {self.interval}')
        start = timestamp
        now = datetime.datetime.timestamp(datetime.datetime.now())*1000
        # estimates time required to complete api request
        requests_required = (now - timestamp) / 60000 / BINANCE_INTERVAL_MINUTES[self.interval] / 1000
        # prevents redundant data requests
        if requests_required < 0.05:
            print(f'{requests_required:.2f} is not enough requests required. No data is being fetched')
            return
        print(f'{requests_required:.2f} requests required. Estimated time is {requests_required/75:.0f} minutes.')
        requests_count = 0
        # requests klines and stores in df until present time
        while start < now:
            bars = client.get_historical_klines(self.ticker, 
                self.interval, str(start), limit=5000)
            bars_df = pd.DataFrame(bars, columns=['Datetime', 'Open', 'High', 
            'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 
            'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore'])

            bars_df['Datetime'] = pd.to_datetime(bars_df['Datetime'].astype(int)/1000, unit='s')
            df = pd.concat([df,bars_df])
            requests_count += 1
            print(f'Progress: {requests_count/requests_required*100:.2f}% - ({requests_count}/{requests_required:.0f})')
            if bars:
                start = int(bars[-1][0]) + 1
            else:
                break

        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'Quote Asset Volume', 'TB Base Volume', 'TB Quote Volume']
        
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)
        df.drop('Close Time', 1)

        df.to_csv(self.dir, index=False)

        return df
    

    def file_exists(self):
        '''tests whether price data file already exists and returns last 
        price date if so, otherwise returns false'''
        dir = self.dir
        # tests file exists
        if os.path.exists(dir):
            df = pd.read_csv(dir)

            # uses yf datetime format
            if self.source == 'yahoo':
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
            
            # uses binance datetime format
            elif self.source == 'binance':
                # if self.interval in ['1d', '1w']:
                #     format_str = "%Y-%m-%d"
                # else:
                #     format_str = "%Y-%m-%d %H:%M:%S"

                try:
                    format_str = "%Y-%m-%d %H:%M:%S"
                    recent_date = datetime.datetime.strptime(df.iloc[-1,0:1].values[0], format_str)
                except:
                    format_str = "%Y-%m-%d"
                    recent_date = datetime.datetime.strptime(df.iloc[-1,0:1].values[0], format_str)

                recent_date = pytz.timezone('UTC').localize(recent_date)

                return datetime.datetime.timestamp(recent_date) * 1000

        else:
            if not os.path.exists(f'Price_data/{self.source}/{self.ticker}'):
                os.makedirs(f'Price_data/{self.source}/{self.ticker}')
            return False


    def import_from_database(self, data_points: int=0):
        '''collects specified price data from database and returns it in
        pandas dataframe'''
        
        dir = self.dir
        # imports csv to df
        df = pd.read_csv(dir, index_col=0, parse_dates=True)
        df.drop(df.columns[[4, 5, 6]], axis=1, inplace=True)
        # uses start date if requested
        if self.start_date is not None:
            # index = df[df['Datetime']==self.start_date].index
            df = df.loc[self.start_date:]
        if self.end_date is not None:
            # index = df[df['Datetime']==self.end_date].index
            df = df.loc[:self.end_date]
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


    def yahoo_fetch_ohlc(self):
        '''adds requested data to price data folder as csv'''
        ticker = self.ticker
        interval = self.interval
        dir = self.dir

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
        while i < len(extrema):
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
        
        trend_data = df.iloc[upper_i:]
        extrema_list += [(None, None)] * (len(trend_data))


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

        axs[0][1].hist(trend_time_data, bins=bin_size(trend_time_data, YF_INTERVAL_MINUTES[self.interval]/60))
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


    def test_reversion(self, sma_length_trend, sma_length_midpoint, data_points=0):
        '''Analyses the parameters of a simple reversion strategy and
        returns statistics and visualisations'''
        # replicated parameters
        upper_multiple = 1200
        lower_multiple = 800
        num_orders = 8
        balance = 1000

        # generates a list of the percentage distance from price levels
        open_position_levels_perc = []
        order_distance = (upper_multiple - lower_multiple) / num_orders
        for num in range(lower_multiple, upper_multiple+1, int(order_distance)):
            open_position_levels_perc.append(num/1000)

        open_position_levels_perc.sort(key=lambda x: abs(1-x))
        
        # generates a list of the prices distance from an sma
        distance_list = [0]
        indexes_and_directions = []
        sma_data = self.sma_data(sma_length_midpoint, data_points).items()
        price_data = self.averaged_price(data_points).values.tolist()
        for i in range(len(price_data)-1):
            sma = next(sma_data)
            open_position_levels = [x*price_data[i][0] for x in open_position_levels_perc]
            # Check if the current price value and the next price value are on different sides of any level
            for level in open_position_levels:
                if price_data[i][-1] < level and price_data[i + 1][-1] > level:
                    # Append the index, the level and the direction "up" to the list
                    indexes_and_directions.append((i, level, "up"))
                    
                elif price_data[i][-1] > level and price_data[i + 1][-1] < level:
                    # Append the index and the direction "down" to the list
                    indexes_and_directions.append((i, level, "down"))
            
            
            if price_data[i][0] > sma[1]:
                diff = price_data[i][2] - sma[1]
                perc_diff = diff/sma[1]
                distance_list.append(perc_diff)
                
            else:
                diff = price_data[i][1] - sma[1]
                perc_diff = diff/sma[1]
                distance_list.append(perc_diff)
            


        levels = find_crossing_indexes(distance_list, open_position_levels_perc)

        # plots the continous percentage difference from an sma of specified length
        fig, axes = plt.subplots()
        axes.set(xlabel='Datetime', 
                 ylabel=f'% difference from {sma_length_midpoint}SMA', 
                 title=f'Mean reversion {self.ticker}{self.interval}')
        x = self.import_from_database(data_points).index
        plt.plot(x, distance_list)
        plt.plot(x, [0 for x in range(len(distance_list))])
        # self.ohlc_chart(smas=[60], data_points=data_points)'''

        trades = csv.writer()
        return distance_list



# update_database()
Pair('PL=F', 'binance').tf_1h.hist_chart()





runtime = time.time()-start_time
print(f'------ Total Runtime: {(time.time()-start_time)/60:.2f} minutes ------')

plt.show()