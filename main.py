import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# Download stock data for MSFTsiCo and Coca-Cola
tickers = ['MSFT', 'AAPL']
data = yf.download(tickers, start='2018-01-01', end='2023-01-01')

# Print the column structure to understand the data format
print(data.columns)

# Select the 'Close' price for both tickers
close_data = data['Close']

# Preview the data
close_data.head()

# Perform cointegration test
score, p_value, _ = coint(close_data['MSFT'], close_data['AAPL'])

print(f'Cointegration test p-value: {p_value}')

# If p-value is low (<0.05), the pairs are cointegrated
if p_value < 0.05:
    print("The pairs are cointegrated.")
else:
    print("The pairs are not cointegrated.")

# Calculate the spread between the two stocks
# close_data['Spread'] = close_data['MSFT'] - close_data['AAPL']

X = sm.add_constant(close_data['AAPL'])
model = sm.OLS(close_data['MSFT'], X).fit()
hedge_ratio = model.params['AAPL']

close_data['Spread'] = close_data['MSFT'] - hedge_ratio * close_data['AAPL']

# Plot the spread
plt.figure(figsize=(10, 6))
plt.plot(close_data.index, close_data['Spread'], label='Spread (MSFT - AAPL)')
plt.axhline(close_data['Spread'].mean(), color='red', linestyle='--', label='Mean')
plt.legend()
plt.title('Spread between MSFT and AAPL')
plt.show()

# Now try new approach with calculating rolling mean/std
# window = 117
# rolling_mean = close_data['Spread'].rolling(window).mean()
# rolling_std = close_data['Spread'].rolling(window).std()
# close_data['Z-Score'] = (close_data['Spread'] - rolling_mean) / rolling_std


# Define z-score to normalize the spread
close_data['Z-Score'] = (close_data['Spread'] - close_data['Spread'].mean()) / close_data['Spread'].std()

# Set thresholds for entering and exiting trades
upper_threshold = 1.18
lower_threshold = .5

# Initialize signals
close_data['Position'] = 0

'''
# Generate signals for long and short positions
close_data['Position'] = np.where(close_data['Z-Score'] > upper_threshold, -1, close_data['Position'])  # Short the spread
close_data['Position'] = np.where(close_data['Z-Score'] < lower_threshold, 1, close_data['Position'])   # Long the spread
close_data['Position'] = np.where((close_data['Z-Score'] < 1) & (close_data['Z-Score'] > -1), 0, close_data['Position'])  # Exit
'''

# Scale position based on z-score strength
# Cap max leverage at +/-1 (optional)
scaling_factor = 3.8  # You can tune this value
close_data['Position'] = -close_data['Z-Score'] / scaling_factor

# Clip to avoid extreme position sizing
close_data['Position'] = close_data['Position'].clip(lower=-1, upper=1)

# Plot z-score and positions
plt.figure(figsize=(10, 6))
plt.plot(close_data.index, close_data['Z-Score'], label='Z-Score')
plt.axhline(upper_threshold, color='red', linestyle='--', label='Upper Threshold')
plt.axhline(lower_threshold, color='green', linestyle='--', label='Lower Threshold')
plt.legend()
plt.title('Z-Score of the Spread with Trade Signals')
plt.show()

# Calculate daily returns
close_data['MSFT_Return'] = close_data['MSFT'].pct_change()
close_data['AAPL_Return'] = close_data['AAPL'].pct_change()

# Strategy returns: long spread means buying MSFT and shorting AAPL
close_data['Strategy_Return'] = close_data['Position'].shift(1) * (close_data['MSFT_Return'] - close_data['AAPL_Return'])

# Cumulative returns
close_data['Cumulative_Return'] = (1 + close_data['Strategy_Return']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(close_data.index, close_data['Cumulative_Return'], label='Cumulative Return from Strategy')
plt.title('Cumulative Returns of Pairs Trading Strategy')
plt.legend()
plt.show()

# Calculate Sharpe Ratio
sharpe_ratio = close_data['Strategy_Return'].mean() / close_data['Strategy_Return'].std() * np.sqrt(252)
print(f'Sharpe Ratio: {sharpe_ratio}')

# Calculate max drawdown
cumulative_max = close_data['Cumulative_Return'].cummax()
drawdown = (cumulative_max - close_data['Cumulative_Return']) / cumulative_max
max_drawdown = drawdown.max()
print(f'Max Drawdown: {max_drawdown}')
