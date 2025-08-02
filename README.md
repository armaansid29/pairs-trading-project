# pairs-trading-project

# 📈 Statistical Arbitrage: Pairs Trading Strategy with Python

This project implements a **Statistical Arbitrage (StatArb)** strategy using the concept of **Pairs Trading**, a market-neutral trading strategy that seeks to profit from temporary divergences in historically correlated asset prices.

Using Python and historical stock data from `yfinance`, this project identifies cointegrated stock pairs, constructs a trading strategy based on mean reversion, and evaluates its performance through simulation.

---

## 🔍 Project Overview

Pairs trading is based on the statistical relationship between two assets. If two stocks move together historically (i.e., they are cointegrated), and a temporary divergence in their spread occurs, a trader can take advantage by:
- Shorting the overperforming stock
- Buying the underperforming stock
- Closing the position when they revert

This strategy is **market-neutral** — it aims to profit regardless of overall market direction.

---

## ⚙️ Technologies & Libraries Used

- **Python 3.9**
- `pandas` – data manipulation
- `yfinance` – fetching historical stock prices
- `numpy` – numerical calculations
- `matplotlib` – data visualization
- `statsmodels` – cointegration testing (ADF test)
- `seaborn` – enhanced plotting

---

## 📊 Strategy Implementation Steps

### 1. **Data Collection**
- Pulled daily historical stock data using the `yfinance` API:
- Example pairs: PEP vs. KO

### 2. **Data Preprocessing**
- Aligned date indices
- Cleaned and normalized price data for accurate spread calculations

### 3. **Cointegration Test**
- Used the **Augmented Dickey-Fuller (ADF)** test to determine if two stocks are cointegrated
- Selected pairs with p-values below 0.05 as candidates

### 4. **Spread Calculation**
- Modeled the relationship using linear regression:
  ```
  spread = stock_A - β * stock_B
  ```
- Calculated z-scores of the spread to standardize entry/exit signals

### 5. **Trading Signal Logic**
- **Enter Position:**
  - Go long/short when z-score crosses ±1.0
- **Exit Position:**
  - Close when z-score reverts to 0

### 6. **Performance Simulation**
- Simulated the hypothetical trades over historical data
- Recorded position values and plotted cumulative returns

---

## 📈 Results & Takeaways

- The strategy effectively identified mean-reverting opportunities between KO and PEP.
- **Cointegration p-value** confirmed statistical validity for pairs trading.
- Simulated trades demonstrated:
  - Positive cumulative return
  - Reasonable reversion-based entry and exit points

⚠️ **Note:** This is a simplified academic simulation. It does not include:
- Transaction costs
- Slippage
- Position sizing
- Risk management or stop-loss logic

---

## 🧠 Key Learnings

- Built end-to-end pipeline for a basic statistical arbitrage strategy
- Applied cointegration testing using real financial time series data
- Implemented signal generation using z-score of residual spread
- Understood core principles behind market-neutral trading strategies

---

## 🖥️ Output Plots

> Add these after generating:

- Price chart with entry/exit points  
- Spread vs. z-score over time  
- Cumulative returns of strategy vs. benchmark  

---

## 🚀 Use the Code

Feel free to use this code and repurpose it however you would like (hopefully you can improve it too). Enjoy!

---

## 📬 Contact

If you'd like to discuss quantitative finance, data-driven strategies, or have feedback, feel free to reach out!
