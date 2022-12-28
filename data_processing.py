import pandas as pd

class Pair:
    def __init__(self, ticker, interval):
        self.ticker = ticker
        self. interval = interval
    
    def __str__(self):
        return self.ticker + self.interval

BTC_USD_15m = Pair('BTC-USD', '15m')

def import_data(ticker, interval, data_points=0):
    '''collects specified price data from database and returns it in
    pandas dataframe'''
    
    file_path = f'Price_data/{ticker}/{interval}.csv'

    # imports csv to df
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.drop(df.columns[[4, 5, 6]], axis=1, inplace=True)
    # shortens dataframe if requested
    if data_points:
        df = df.iloc[len(df)-data_points:len(df)]
    return df


def averaged_price(df):
    '''mutates df to have column of averaged open, high, low and close
    called "price"'''

    average_price = []
    # averages ohlc for each row and adds it to list
    rows = df.itertuples()
    for row in rows:
        sum_price = float(row[1]) + float(row[2]) + float(row[3]) + float(row[4])
        average_price.append(sum_price/4)
    
    # adds column with list
    df['Price'] = average_price
    return df


def sma_data(df, length):
    '''from price df adds a column of SMA with specified length'''
    
    # determines if Price column must be added
    if 'Price' in df.columns:
        None
    else:
        averaged_price(df)

    # returns moving average
    return df['Price'].rolling(length).mean()