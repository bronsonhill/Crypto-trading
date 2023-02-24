from binance import Client
from binance.enums import *
from binance.helpers import round_step_size
import datetime
import pandas as pd
import json
import pytz
from time import sleep
from pushover_complete import PushoverAPI

CLIENT = Client(
            api_key='FuaLBWg3iJCPTTQrU1yewim305sUVvZwOzO4Xau75JLHP0lTlpK9V7bdbPSBZOwF', 
            api_secret='b8mt4w5PKu0gIvDoWHWSpGxdj8dWzesQ4RFCufcHf7l1D6LFMwQN2BAzyqN3lcTI'
    )


api = PushoverAPI("ai2w3cjw17p8esx54eufpdrm9g82ej")

# parameters
MIDPOINT = 10000
UPPER_RANGE = 10005
LOWER_RANGE = 9995
SYMBOL = 'BUSDDAI'
BASE_ASSET = 'BUSD'
QUOTE_ASSET = 'DAI'
ONLY_MAKER = True


order_type = {'id': 'position'}


def refresh_key_data():
    '''refreshes balances and orderbook'''
    global base_asset_balance
    base_asset_balance = CLIENT.get_asset_balance(asset=BASE_ASSET)
    global quote_asset_balance
    quote_asset_balance = CLIENT.get_asset_balance(asset=QUOTE_ASSET)
    global total_free_bal
    total_free_bal = float(base_asset_balance['free']) + float(quote_asset_balance['free'])
    total_free_bal = 1000
    global total_bal
    total_bal = total_free_bal + float(base_asset_balance['free'])
    global orderbook
    orderbook = CLIENT.get_orderbook_tickers(SYMBOL)
    return

refresh_key_data()


def position_levels(midpoint=MIDPOINT, lower_range=LOWER_RANGE, upper_range=UPPER_RANGE):
    '''calculates positon order levels and returns a list of
    them in descending order of their distance from the midpoint'''
    level_list = []
    for level in range(LOWER_RANGE, UPPER_RANGE+1, 1):
        level_list.append(level)

    # sorts levels according to their distance from the midpoint
    level_list.sort(key=lambda x: abs(x-10000))
    return level_list


def log_trade(exitOrder, orderId=None, order=False):
    '''saves given order (of orderId) to open_orders.json'''
    # retrieves specified order details when required
    if not order:
        order = CLIENT.get_order(
            symbol=SYMBOL, 
            orderId=orderId)

    order['exitOrder'] = exitOrder
    # gets existing data
    with open("open_orders.json", 'r') as fp:
        open_orders = json.load(fp)
    open_orders.append(order)
    # saves new data
    with open("open_orders.json", 'w') as fp:
        json.dump(open_orders, fp, indent=4)

    return


def initialise_position_orders():
    '''Places position orders to setup reversion strategy
    1. Gets open position order levels
    2. Checks a position order does not currently exist at a given level
    3. Determines the correct direction of position order
    4. Checks the order price is within the orderbook
    5. Skips the price level if there is a pending exit order'''
    
    # assesses open position orders and records their price levels
    with open('open_orders.json', 'r') as fp:
        open_orders = json.load(fp)

    open_order_prices = []
    buy_exit_count = 0
    sell_exit_count = 0
    for order in open_orders:
        if not order['exitOrder']:
            open_order_prices.append(order['price'])
        else:
            if order['side'] == 'BUY':
                buy_exit_count += 1
            else:
                sell_exit_count += 1

    # then places maker orders around range parameters when required
    position_level_list = position_levels()
    for price in position_level_list:
        refresh_key_data()
        price = price/10000
        # checks if there is already a position order open at price level
        if price not in [round_step_size(x, 0.0001) for x in open_order_prices]:
            # records time to retrieve and save order details
            start_time = int(datetime.datetime.timestamp(datetime.datetime.now(pytz.timezone('UTC'))) * 1000)
            # quantity/size calculation
            quantity = round_step_size(int(total_free_bal/(len(position_level_list)-1)), 0.01)
            # sets sell orders above midpoint and disallows a taker order
            if price > MIDPOINT//10000 and price > float(orderbook['bidPrice']):
                # ensures balance is available
                if float(base_asset_balance['free']) >= quantity:
                    # skips level if there is a pending exit order
                    if buy_exit_count > 0:
                        buy_exit_count -= 1
                    else:
                        # creates and submits order
                        CLIENT.create_order(
                            symbol=SYMBOL,
                            side=SIDE_SELL,
                            type=ORDER_TYPE_LIMIT,
                            timeInForce=TIME_IN_FORCE_GTC,
                            quantity=quantity,
                            price=price
                            )
                        # logs new trade
                        log_trade(
                            exitOrder=False, 
                            order=CLIENT.get_all_orders(
                                symbol=SYMBOL, 
                                startTime=start_time
                            )[0]
                    )
                else:
                    print('INSUFFICIENT BALANCE')
                    return
        
            # sets buy orders below midpoint and disallows a taker order
            elif price < MIDPOINT//10000 and price < float(orderbook['askPrice']):
                # ensures balance is available
                if float(quote_asset_balance['free']) >= quantity:
                    # skips level if there is a pending exit order
                    if sell_exit_count > 0:
                        sell_exit_count -= 1
                    else:
                        CLIENT.create_order(
                            symbol=SYMBOL,
                            side=SIDE_BUY,
                            type=ORDER_TYPE_LIMIT,
                            timeInForce=TIME_IN_FORCE_GTC,
                            quantity=quantity,
                            price=price
                            )
                        # logs new trade
                        log_trade(
                            exitOrder=False, 
                            order=CLIENT.get_all_orders(
                                symbol=SYMBOL, 
                                startTime=start_time
                            )[0]
                        )
                else:
                    print('INSUFFICIENT BALANCE')
                    return
    return


def standby(refresh_time=1):
    '''returns orders that have been filled'''

    while True:
        sleep(refresh_time)
        print('Watching orders')

        with open("open_orders.json", 'r') as fp:
            open_orders = json.load(fp)
        
        i = 0
        for order in open_orders:
            current_order_details = CLIENT.get_order(
                symbol=SYMBOL,
                orderId=order['orderId'])
            # removes from open orders json
            if current_order_details['status'] == 'FILLED':
                print('Order filled')
                api.send_message("uw1rwwrqhzfi9hai19gdt9bkgjq8as", SYMBOL, device="iPhone", title=f"{order['side']} order filled: {order['origQty']} @{order['price']}")
                open_orders.pop(i)
                with open("open_orders.json", 'w') as fp:
                    json.dump(open_orders, fp, indent=4)
                # places exit order if recently filled order is not an exit order
                if not order['exitOrder']:
                    place_exit_order(order)
                    with open("trade_log.json", 'r') as fp:
                        trade_log = json.load(fp)
                    trade_log.append(current_order_details)
                    with open("trade_log.json", 'w') as fp:
                        json.dump(trade_log, fp, indent=4)
                    
                # otherwise places position order
                else:
                    initialise_position_orders()
                    with open("trade_log.json", 'r') as fp:
                        trade_log = json.load(fp)
                    trade_log.append(current_order_details)
                    with open("trade_log.json", 'w') as fp:
                        json.dump(trade_log, fp, indent=4)
                    
                    
            i += 1
        print('...')
                

def place_exit_order(filled_order):
    '''places an exit trade according to the inputted filled trade'''
    
    # determines side of exit trade
    if filled_order['side'] == SIDE_BUY:
        side = SIDE_SELL
    else:
        side = SIDE_BUY
    
    price = MIDPOINT/10000
    
    # records time to log new order 
    start_time = int(datetime.datetime.timestamp(datetime.datetime.now(pytz.timezone('UTC'))) * 1000)
    # determines quantity of exit trade
    quantity = float(filled_order['origQty'])

    CLIENT.create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=price
            )
    
    log_trade(
        exitOrder=True, 
        order=CLIENT.get_all_orders(
            symbol=SYMBOL, 
            startTime=start_time
            )[0]
        )

    return 

initialise_position_orders()
standby()