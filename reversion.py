from binance import Client
from binance.enums import *
from binance.helpers import round_step_size
import datetime
import pandas as pd
from time import sleep

CLIENT = Client(
            api_key='FuaLBWg3iJCPTTQrU1yewim305sUVvZwOzO4Xau75JLHP0lTlpK9V7bdbPSBZOwF', 
            api_secret='b8mt4w5PKu0gIvDoWHWSpGxdj8dWzesQ4RFCufcHf7l1D6LFMwQN2BAzyqN3lcTI'
    )

# parameters
MIDPOINT = 10000
UPPER_RANGE = 10005
LOWER_RANGE = 9995
SYMBOL = 'BUSDDAI'
BASE_ASSET = 'BUSD'
QUOTE_ASSET = 'DAI'


order_type = {'id': 'position'}


def refresh_key_data():
    '''refreshes balances and orderbook'''
    global base_asset_balance
    base_asset_balance = CLIENT.get_asset_balance(asset=BASE_ASSET)
    global quote_asset_balance
    quote_asset_balance = CLIENT.get_asset_balance(asset=QUOTE_ASSET)
    global total_bal
    total_bal = float(base_asset_balance['free']) + float(quote_asset_balance['free'])
    global orderbook
    orderbook = CLIENT.get_orderbook_tickers(SYMBOL)
    return

refresh_key_data()

def calculate_initial_levels(midpoint=MIDPOINT, lower_range=LOWER_RANGE, upper_range=UPPER_RANGE):
    '''calculates the initial positon order levels and returns a list of
    them in descending order of their distance from the midpoint'''
    level_list = []
    for level in range(LOWER_RANGE, UPPER_RANGE, 1):
        level_list.append(level)
    
    level_list.sort(key=lambda x: abs(x-10000))
    return level_list


def initialise_orders():
    '''Places orders to setup reversion strategy'''
    # begins by assessing open orders
    existing_orders = CLIENT.get_all_orders(symbol=SYMBOL)
    order_df = pd.DataFrame(columns=['orderId', 'side', 'price', 'type'])
    # for order in existing_orders:
        

    # then places maker orders around range parameters
    for price in calculate_initial_levels():
        price = price/10000
        # if price in order_df['price'] and 
        quantity = int(total_bal/(UPPER_RANGE-LOWER_RANGE))
        quantity = abs(price - 1)
        
        # sets sell orders above midpoint and disallows a taker order
        if price > MIDPOINT//10000 and price >= float(orderbook['askPrice']):
            if base_asset_balance >= quantity:
                CLIENT.create_order(
                    symbol=SYMBOL,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=quantity,
                    price=price
                    )
            else: 
                return
        
        # sets buy orders below midpoint and disallows a taker order
        elif price < MIDPOINT//10000 and price <= float(orderbook['bidPrice']):
            if quote_asset_balance >= quantity:
                CLIENT.create_order(
                    symbol=SYMBOL,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=quantity,
                    price=price
                    )
            else:
                return
    return


def standby(startTime=0, refresh_time=1):
    '''returns orders that have been filled'''
    
    while True:
        sleep(refresh_time)

        all_orders = CLIENT.get_all_orders(
            symbol=SYMBOL, 
            startTime=startTime)
        
        for order in all_orders:
            order_id = order['orderId']
            if order['status'] == 'FILLED':
                return order


def place_exit_order():
    '''places an exit trade according to the inputted filled trade'''

    filled_order = standby()
    
    # determines side of exit trade
    if filled_order['side'] == SIDE_BUY:
        side = SIDE_SELL
    else:
        side = SIDE_BUY
    
    # determines price of exit trade
    if float(filled_order['price']) == 1.0001:
        price = 0.9999
    elif float(filled_order['price']) == 0.9999:
        price = 1.0001
    elif float(filled_order['price']) > 1.0001:
        price = 1
    elif float(filled_order['price']) < 0.9999:
        price = 1
    
    # determines quantity of exit trade
    quantity = float(filled_order['quantity'])

    CLIENT.create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=price
            )

    initialise_orders()
    standby(int(filled_order['Time']))

    return 