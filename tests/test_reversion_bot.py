import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reversion import ReversionBot

SIDE_BUY = 'BUY'

class TestReversionBot(unittest.TestCase):

    @patch('reversion.Client')
    @patch('reversion.PushoverAPI')
    def setUp(self, MockPushoverAPI, MockClient):
        self.mock_client = MockClient()
        self.mock_api = MockPushoverAPI()
        self.bot = ReversionBot(api_key='test_key', api_secret='test_secret', pushover_key='test_pushover_key')

    def test_setup_logging(self):
        with self.assertLogs(level='INFO') as log:
            self.bot.setup_logging()
            self.assertIn('INFO', log.output[0])

    def test_load_config(self):
        self.bot.load_config()
        self.assertEqual(self.bot.midpoint, 10000)
        self.assertEqual(self.bot.upper_range, 10005)
        self.assertEqual(self.bot.lower_range, 9995)
        self.assertEqual(self.bot.symbol, 'USDCUSDT')
        self.assertEqual(self.bot.base_asset, 'USDC')
        self.assertEqual(self.bot.quote_asset, 'USDT')
        self.assertTrue(self.bot.only_maker)

    @patch('reversion.Client.get_asset_balance')
    @patch('reversion.Client.get_orderbook_tickers')
    def test_refresh_key_data(self, mock_get_orderbook_tickers, mock_get_asset_balance):
        mock_get_asset_balance.side_effect = [
            {'free': '1000'},  # base_asset_balance
            {'free': '2000'}   # quote_asset_balance
        ]
        mock_get_orderbook_tickers.return_value = {'bidPrice': '1.0', 'askPrice': '1.1'}

        self.bot.refresh_key_data()
        self.assertEqual(self.bot.total_free_bal, 3000.0)
        self.assertEqual(self.bot.orderbook['bidPrice'], '1.0')
        self.assertEqual(self.bot.orderbook['askPrice'], '1.1')

    @patch('reversion.json.load')
    @patch('builtins.open')
    def test_load_open_orders(self, mock_open, mock_json_load):
        mock_json_load.return_value = [{'orderId': '1', 'price': '1.0'}]
        open_orders = self.bot.load_open_orders()
        self.assertEqual(open_orders, [{'orderId': '1', 'price': '1.0'}])

    @patch('reversion.Client.create_order')
    @patch('reversion.Client.get_all_orders')
    def test_place_order(self, mock_get_all_orders, mock_create_order):
        mock_get_all_orders.return_value = [{'orderId': '1', 'price': '1.0'}]
        self.bot.place_order(SIDE_BUY, 1, '1.0', 1234567890)
        mock_create_order.assert_called_once()
        mock_get_all_orders.assert_called_once()
    
    # Additional tests for other methods...

if __name__ == '__main__':
    unittest.main()