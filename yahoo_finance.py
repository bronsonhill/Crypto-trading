import datetime
import yfinance
import pandas as pd
import os
from pytz import UTC


HISTORY_LIMIT = {'1m': 30, '2m': 60, '5m': 60, '15m': 60, '30m': 60, 
'1h': 730, '1d': 730, '1wk': 10000, '1mo': 300}

REQUEST_LIMIT = {'1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60, 
'1h': 730, '1d': 730, '1wk': 10000, '1mo': 10000}


def fetch_ohlc_data(ticker, interval, start, end):
    '''fetches ohlc data for ticker for interval and start and end time'''

    ticker = yfinance.Ticker(ticker)
    df = ticker.history(start=start, end=end, interval=interval)

    return df


def file_exists(dir):
    '''tests whether price data file already exists and returns last 
    price date if so, otherwise returns false'''
    
    # tests file exists
    if os.path.exists(dir):
        file = pd.read_csv(dir)

        # determines the required datetime format
        for elem in ['1d', '1wk', '1mo']:
            if elem in dir:
                format_str = "%Y-%m-%d"
                break
            else:
                format_str = "%Y-%m-%d %H:%M:%S%z"

        # gives last price time
        recent_date = datetime.datetime.strptime(
            file.iloc[1:2,0:1].values[0][0],format_str) \
            + datetime.timedelta(hours=16, minutes=1)

        return recent_date
    
    else:
        return False



def compile_data(ticker, interval):
    '''adds requested data to price data folder as csv'''
    
    # save directory
    dir = f'Price_data/{ticker}/{interval}.csv'

    # creates ticker folder if required
    if not os.path.exists(f'Price_data/{ticker}'):
        os.makedirs(f'Price_data/{ticker}')

    # sets start date of request according to whether file exists or not
    today = datetime.datetime.now()
    today = UTC.localize(today)
    
    exist_bool = file_exists(dir)
    if exist_bool:
        start = exist_bool
    
    # sets start date to oldest date allowed by yfinance
    else:
        try:
            start = datetime.datetime.now() - datetime.timedelta(
            days=HISTORY_LIMIT[interval])
            start = UTC.localize(start)

        except:
            print(f'Try interval: {[interval for interval in HISTORY_LIMIT.keys()]}')
            return
    
    df = pd.DataFrame()
    
    # makes request for data and appends to pandas dataframe
    end = start
    while end < today:
        end = start + datetime.timedelta(days=REQUEST_LIMIT[interval])
        if not df.empty:
            df = pd.concat([df, fetch_ohlc_data(ticker, interval, start, end)])
        else:
            df = pd.DataFrame(fetch_ohlc_data(ticker, interval, start, end))
        start = end
    
    # either appends dataframe to existing csv or creates new one
    if exist_bool:
        df.to_csv(dir, mode='a', index=True, header=False)
    else:
        df.to_csv(dir)
    
    return


def update_ticker_data(ticker):
    '''compiles data for all intervals for a pair'''

    # makes request for every interval
    for interval in HISTORY_LIMIT.keys():
        print(f'Compiling {interval} data for {ticker}.')
        compile_data(ticker, interval)
    print('Complete')

    return


update_ticker_data('GC=F')