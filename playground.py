import pandas as pd
import numpy as np
import datetime
import json
import pytz
from binance import Client
import os
'''
client = Client(
            api_key='FuaLBWg3iJCPTTQrU1yewim305sUVvZwOzO4Xau75JLHP0lTlpK9V7bdbPSBZOwF', 
            api_secret='b8mt4w5PKu0gIvDoWHWSpGxdj8dWzesQ4RFCufcHf7l1D6LFMwQN2BAzyqN3lcTI'
    )


open_orders = []
i = 0
order_details = client.get_all_orders(symbol='BUSDDAI')
for order in order_details:
     if order['status'] == 'NEW':
          open_orders.append(order)
          i += 1
print(i)
with open('open_orders.json', 'w') as fp:
     json.dump(open_orders, fp, indent=4)'''


from pushover_complete import PushoverAPI

# Create a PushoverAPI object with your API token
api = PushoverAPI("ai2w3cjw17p8esx54eufpdrm9g82ej")

# Send a message to a user or group
api.send_message("uw1rwwrqhzfi9hai19gdt9bkgjq8as", "This is a test notification", device="iPhone", title="Hello from Python")