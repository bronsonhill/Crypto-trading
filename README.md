# crypto-trading
A repository for my crypto-trading experimentation. This repository can be divided into four main areas of functionality; data collection and storage, data analysis, backtesting and strategy execution.

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
- 
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

