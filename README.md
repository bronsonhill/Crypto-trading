# crypto-trading
A repository for my crypto-trading experimentation. This repository can be divided into four main areas of functionality; data collection and storage, data analysis, backtesting and strategy execution.

## Demonstration

### Collect ohlc price data on multiple time from from Binance or Yahoo finance
```Python
Pair('PL=F', 'yahoo').update_data()
```
![Screenshot 2025-01-03 at 9 19 21 am](https://github.com/user-attachments/assets/70362241-efa5-420a-bed9-50152e852dba)

### Chart price data with customisable indicators
```Python
Pair('PL=F', 'yahoo').tf_1m.ohlc_chart(True, [10], [], 300)
```
![Screenshot 2025-01-03 at 9 18 01 am](https://github.com/user-attachments/assets/43adc473-d889-4da7-85ec-a57b544e6f09)
### Analyse distribution of trend movements by percentage change and time
```Python
Pair('PL=F', 'yahoo').tf_1m.hist_chart(sma_length=10)
```
![Screenshot 2025-01-03 at 9 15 51 am](https://github.com/user-attachments/assets/c958e0e0-3aad-4a63-bac7-f5b31f5b80af)

### Analyse the correlation between trend percentage change and time
```Python
Pair('PL=F', 'yahoo').tf_1m.scatter_chart(sma_length=10)
```
![Screenshot 2025-01-03 at 9 16 26 am](https://github.com/user-attachments/assets/bffb2490-fe5a-4a18-914f-0c4bb7b93f48)
![Screenshot 2025-01-03 at 9 20 30 am](https://github.com/user-attachments/assets/d6748af7-0ab9-43d5-96ef-0ea66d704003)

### Arbitrage trade around a fair price
```Python
bot = ReversionBot(api_key=api_key, api_secret=api_secret, pushover_key=pushover_key)
bot.initialise_position_orders()
bot.standby()
```
### Recieve mobile push notifications when trades are made
<img src="https://github.com/user-attachments/assets/a73c6750-8bdd-42cd-9625-f4bcf28da231" alt="drawing" width="400"/>

---
# Development notes
## Data collection and storage
The current state of the price database is trivial; a folder of enormous csv files. It ought to be converted into an SQL database.

## Data analysis
Current functionality:
- display OHLC data
- trend definition: the price extrema in intervals defined by derivatve of MA
- display the distribution of trend price-changes, time-changes and their correlations
- search for abitrage opportunities

Desired functionality:
- analyse pair: applies all analyses and displays the results with matplotlib and outputs results file
- analyse pairs: applies cross-pair analyses and displays the results with matplotlib and outputs results file
- time analysis: search for predictability in price change and volatility at particular times of day, week, month and year
- introduce alternate datapoints: sentiment and basic indicators such as RSI and bollinger bands
- clean data: categorise datapoints where data is missing or exchange is offline


## Backtesting
Current functionality:
- WIP

Desired functionality:
- to run strategy execution of historic OHLC data and output trades.json and stats.json
- basic stats includes win/loss ratio, profit/loss ratio, average win, average loss, number of trades, variance of these numbers
- also alpha, beta, information ratio, maximum adverse excursion, average true range, skewness, kurtosis, ulcer index and conditional value at risk

## Strategy execution
Current functionality:
- manage a simple algorithm buying and selling around a stable midpoint
- mobile notifications
- output trades.json
Desired functionality:
- analyse trades.json
- execute algorithm with variable midpoint
- risk management
- execute arbitrage algorithm

