import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# download stock data for Pepsi & Coca Cola
tickers = ['PEP', 'KO']
data = yf.download(tickers, start='2015-01-01', end='2024-01-01')

# testing to make sure data comes in right (had lots of issues with this lol)
print(data.columns)

# select the 'Close' price for both tickers
close_data = data['Close']
close_data.head()

# perform cointegration test
# this is how you can statistically tell if the stocks are truly related --> the strategy lies on this assumption
score, p_value, _ = coint(close_data['PEP'], close_data['KO'])

# print(f'Cointegration test p-value: {p_value}')

# if p-value < 0.05, the pairs are cointegrated (p-value is just probability that it can randomly happen)
if p_value < 0.05:
    print("The pairs are cointegrated.")
else:
    print("The pairs are not cointegrated.")

# now calculate and plot the spread between the two stocks
close_data['Spread'] = close_data['PEP'] - close_data['KO']
plt.figure(figsize=(10, 6))
plt.plot(close_data.index, close_data['Spread'], label='Spread (PEP - KO)')
plt.axhline(close_data['Spread'].mean(), color='red', linestyle='--', label='Mean')
plt.legend()
plt.title('Spread between PEP and KO')
plt.show()

# use z-score to normalize the spread, helps with consistent thresholds
close_data['Z-Score'] = (close_data['Spread'] - close_data['Spread'].mean()) / close_data['Spread'].std()

# set the thresholds for entering and exiting trades (change this around for different Sharpe ratios)
upper_threshold = 1.75
lower_threshold = 1.75

# give signals for the short and long positions
close_data['Position'] = 0
close_data['Position'] = np.where(close_data['Z-Score'] > upper_threshold, -1, close_data['Position'])  # Short the spread
close_data['Position'] = np.where(close_data['Z-Score'] < lower_threshold, 1, close_data['Position'])   # Long the spread
close_data['Position'] = np.where((close_data['Z-Score'] < 1) & (close_data['Z-Score'] > -1), 0, close_data['Position'])  # Exit

# plot z-score and positions
plt.figure(figsize=(10, 6))
plt.plot(close_data.index, close_data['Z-Score'], label='Z-Score')
plt.axhline(upper_threshold, color='red', linestyle='--', label='Upper Threshold')
plt.axhline(lower_threshold, color='green', linestyle='--', label='Lower Threshold')
plt.legend()
plt.title('Z-Score of the Spread with Trade Signals')
plt.show()

# calculate daily returns
close_data['PEP_Return'] = close_data['PEP'].pct_change()
close_data['KO_Return'] = close_data['KO'].pct_change()

# strategy returns: long spread means buying PEP and shorting KO
close_data['Strategy_Return'] = close_data['Position'].shift(1) * (close_data['PEP_Return'] - close_data['KO_Return'])

# cumulative returns and plot
close_data['Cumulative_Return'] = (1 + close_data['Strategy_Return']).cumprod()
plt.figure(figsize=(10, 6))
plt.plot(close_data.index, close_data['Cumulative_Return'], label='Cumulative Return from Strategy')
plt.title('Cumulative Returns of Pairs Trading Strategy')
plt.legend()
plt.show()

# now backtest to see how well the strategy did
# calculate the Sharpe Ratio and max drawdown
sharpe_ratio = close_data['Strategy_Return'].mean() / close_data['Strategy_Return'].std() * np.sqrt(252)
print(f'Sharpe Ratio: {sharpe_ratio}')
cumulative_max = close_data['Cumulative_Return'].cummax()
drawdown = (cumulative_max - close_data['Cumulative_Return']) / cumulative_max
max_drawdown = drawdown.max()
print(f'Max Drawdown: {max_drawdown}')
