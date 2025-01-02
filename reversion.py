import os
from dotenv import load_dotenv
import logging
import datetime
import pytz
import json
from binance import Client
from binance.enums import *
from binance.helpers import round_step_size
from pushover_complete import PushoverAPI
from time import sleep

class ReversionBot:
    def __init__(self, api_key, api_secret, pushover_key):
        self.client = Client(api_key, api_secret)
        self.api = PushoverAPI(pushover_key)
        self.setup_logging()
        self.load_config()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def load_config(self):
        # Load parameters from a config file or environment variables
        self.midpoint = 10000
        self.upper_range = 10005
        self.lower_range = 9995
        self.symbol = 'USDCUSDT'
        self.base_asset = 'USDC'
        self.quote_asset = 'USDT'
        self.only_maker = True
    
    def refresh_key_data(self):
        try:
            self.base_asset_balance = self.client.get_asset_balance(asset=self.base_asset)
            self.quote_asset_balance = self.client.get_asset_balance(asset=self.quote_asset)
            self.total_free_bal = float(self.base_asset_balance['free']) + float(self.quote_asset_balance['free'])
            self.orderbook = self.client.get_orderbook_tickers(symbol=self.symbol)
        except Exception as e:
            logging.error(f"Error refreshing key data: {e}")
    
    def position_levels(self):
        level_list = list(range(self.lower_range, self.upper_range + 1))
        level_list.sort(key=lambda x: abs(x - self.midpoint))
        return level_list
    
    def log_trade(self, exit_order, order=None):
        try:
            if not order:
                order = self.client.get_order(symbol=self.symbol, orderId=exit_order['orderId'])
            order['exitOrder'] = exit_order
            with open("open_orders.json", 'r+') as fp:
                open_orders = json.load(fp)
                open_orders.append(order)
                fp.seek(0)
                json.dump(open_orders, fp, indent=4)
        except Exception as e:
            logging.error(f"Error logging trade: {e}")
    
    def initialise_position_orders(self):
        '''Places position orders to setup reversion strategy'''
        
        open_orders = self.load_open_orders()

        open_order_prices, buy_exit_count, sell_exit_count = self.assess_open_orders(open_orders)

        position_level_list = self.position_levels()
        total_levels = len(position_level_list) - 1
        for price in position_level_list:
            self.refresh_key_data()
            adjusted_price = price / 10000

            if adjusted_price in [round_step_size(x, 0.0001) for x in open_order_prices]:
                continue

            start_time = int(datetime.datetime.timestamp(datetime.datetime.now(pytz.timezone('UTC'))) * 1000)
            quantity = round_step_size(int(self.total_free_bal / total_levels), 0.01)

            if adjusted_price > self.midpoint / 10000 and adjusted_price > float(self.orderbook['bidPrice']):
                if float(self.base_asset_balance['free']) < quantity:
                    logging.error('INSUFFICIENT BALANCE for SELL order at price {}'.format(adjusted_price))
                    return
                if buy_exit_count > 0:
                    buy_exit_count -= 1
                    continue
                self.place_order(SIDE_SELL, quantity, adjusted_price, start_time)
            
            elif adjusted_price < self.midpoint / 10000 and adjusted_price < float(self.orderbook['askPrice']):
                if float(self.quote_asset_balance['free']) < quantity:
                    logging.error('INSUFFICIENT BALANCE for BUY order at price {}'.format(adjusted_price))
                    return
                if sell_exit_count > 0:
                    sell_exit_count -= 1
                    continue
                self.place_order(SIDE_BUY, quantity, adjusted_price, start_time)

        return

    def load_open_orders(self):
        try:
            with open('open_orders.json', 'r') as fp:
                return json.load(fp)
        except Exception as e:
            logging.error(f"Error loading open orders: {e}")
            return []

    def assess_open_orders(self, open_orders):
        open_order_prices = []
        buy_exit_count = 0
        sell_exit_count = 0
        for order in open_orders:
            if not order.get('exitOrder', False):
                open_order_prices.append(order.get('price'))
            else:
                if order.get('side') == 'BUY':
                    buy_exit_count += 1
                else:
                    sell_exit_count += 1
        return open_order_prices, buy_exit_count, sell_exit_count

    def place_order(self, side, quantity, price, start_time):
        try:
            self.client.create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                price=price
            )
            order = self.client.get_all_orders(
                symbol=self.symbol, 
                startTime=start_time
            )[0]
            self.log_trade(exitOrder=False, order=order)
            logging.info(f"Placed {side} order: {quantity} @ {price}")
        except Exception as e:
            logging.error(f"Error placing {side} order at {price}: {e}")
    
    def standby(self, refresh_time=1):
        while True:
            sleep(refresh_time)
            logging.info('Watching orders')

            open_orders = self.load_open_orders()

            for i, order in enumerate(open_orders):
                try:
                    current_order_details = self.client.get_order(
                        symbol=self.symbol,
                        orderId=order['orderId']
                    )
                except Exception as e:
                    logging.error(f"Error fetching order details for Order ID {order['orderId']}: {e}")
                    continue

                if current_order_details.get('status') == 'FILLED':
                    logging.info('Order filled')
                    try:
                        self.api.send_message(
                            "uw1rwwrqhzfi9hai19gdt9bkgjq8as",
                            self.symbol,
                            device="iPhone",
                            title=f"{order['side']} order filled: {order['origQty']} @{order['price']}"
                        )
                    except Exception as e:
                        logging.error(f"Error sending Pushover message: {e}")

                    open_orders.pop(i)
                    self.update_open_orders(open_orders)

                    if not order.get('exitOrder', False):
                        self.place_exit_order(order)
                    else:
                        self.initialise_position_orders()

                    self.update_trade_log(current_order_details)

            logging.info('...')
    
    def update_open_orders(self, open_orders):
        try:
            with open("open_orders.json", 'w') as fp:
                json.dump(open_orders, fp, indent=4)
            logging.info("Updated open_orders.json")
        except Exception as e:
            logging.error(f"Error updating open_orders.json: {e}")

    def update_trade_log(self, order_details):
        try:
            with open("trade_log.json", 'r+') as fp:
                trade_log = json.load(fp)
                trade_log.append(order_details)
                fp.seek(0)
                json.dump(trade_log, fp, indent=4)
            logging.info("Updated trade_log.json with new order details")
        except FileNotFoundError:
            logging.warning("trade_log.json not found. Creating a new one.")
            try:
                with open("trade_log.json", 'w') as fp:
                    json.dump([order_details], fp, indent=4)
                logging.info("Created trade_log.json with initial order details")
            except Exception as e:
                logging.error(f"Error creating trade_log.json: {e}")
        except Exception as e:
            logging.error(f"Error updating trade_log.json: {e}")
    
    def place_exit_order(self, filled_order):
        '''places an exit trade according to the inputted filled trade'''
        
        # determines side of exit trade
        if filled_order['side'] == SIDE_BUY:
            side = SIDE_SELL
        else:
            side = SIDE_BUY
        
        price = self.midpoint/10000
        
        # records time to log new order 
        start_time = int(datetime.datetime.timestamp(datetime.datetime.now(pytz.timezone('UTC'))) * 1000)
        # determines quantity of exit trade
        quantity = float(filled_order['origQty'])

        self.client.create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                price=price
                )
        
        self.log_trade(
            exitOrder=True, 
            order=self.client.get_all_orders(
                symbol=self.symbol, 
                startTime=start_time
                )[0]
            )

        return 


load_dotenv()

if __name__ == "__main__":
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    pushover_key = os.getenv('PUSHOVER_KEY')
    
    bot = ReversionBot(api_key=api_key, api_secret=api_secret, pushover_key=pushover_key)
    bot.initialise_position_orders()
    bot.standby()