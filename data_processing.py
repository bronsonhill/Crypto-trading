import pandas as pd

def import_data(file_name):
    '''collects specified price data from database and returns df'''
    
    # imports csv to df
    df = pd.read_csv(file_name)
    df.drop(df.columns[[5, 6, 7, 8]], axis=1, inplace=True)

    return df


def averaged_price(df):
    '''mutates df to have column of averaged open, high, low and close'''
    average_price = []

    rows = df.itertuples()
    for row in rows:
        sum_price = float(row[2]) + float(row[3]) + float(row[4]) + float(row[5])
        average_price.append(sum_price/4)
    
    df['Price'] = average_price
    return df


def add_sma(df, length):
    '''from price df adds a column of SMA with specified length'''
    df[f'{length}SMA'] = df['Price'].rolling(length).mean()
    return df