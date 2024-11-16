# Adaptive Price-Drop Investment Strategy

This project implements and tests an adaptive investment strategy that modifies its portfolio weighting based on significant price drops across assets. The core strategy, called **Adaptive Price-Drop (APD)**, reallocates investments dynamically to prioritize stocks that experience rare price decreases relative to their recent performance. The project also includes implementations of a standard **Dollar-Cost Averaging (DCA)** strategy, inflation adjustments, and various investment analytics.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [How it Works](#how-it-works)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project simulates and analyzes the Adaptive Price-Drop (APD) strategy, comparing it to a Dollar-Cost
Averaging (DCA) strategy. Using historical stock data, the APD strategy evaluates the percentile-based
price change of each stock, re-weighting investment allocations to target stocks experiencing unusual
price drops.

### Key Objectives:
- Test and analyze the performance of adaptive vs. traditional investment strategies.
- Use historical CPI data to adjust prices for inflation.
- Enable flexible strategy configuration and easy-to-interpret results for financial analysis.

## Features

- **Adaptive Price-Drop (APD) Strategy**: Rebalances investments based on the relative percentile of
price changes across stocks.
- **Dollar-Cost Averaging (DCA)**: Compares the APD performance with the classic DCA method.
- **Inflation Adjustment**: Adjusts historical prices for inflation using monthly CPI data.
- **Simulation and Analysis**: Generates investment simulations and returns key performance metrics.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/dtilley/quantitative_finance_toolkit
    cd quantitative_finance_toolkit
    ```

2. **Install required packages**:
    - Using `pip`:
      ```bash
      pip install -r requirements.txt
      ```
    - Required packages include:
      - `numpy`
      - `pandas`
      - `yfinance` for stock data
      - `scikit-learn` for machine learning models
      - `scipy` and `matplotlib` for data processing and plotting

3. **Download Historical CPI Data** (optional): Follow the instructions in the `get_cpi()` function to pull CPI data, or use the default CPI dataset included.

## Usage

1. **Initialize Stocks**:
   Use the `Stock` class to create stock objects for each ticker symbol and download their historical data.

2. **Create an Investor**:
   The `Investor` class simulates different strategies, cash flows, and investment periods.

3. **Run Adaptive Price-Drop (APD) Simulation**:
   Using `Adaptive_Price_Drop`, simulate an investment strategy that dynamically adjusts based on price-drop percentiles.

4. **Analyze Results**:
   Generate and interpret results with built-in data analysis functions, including inflation-adjusted returns, gain/loss evaluations, and kernel density estimations for price changes.

### Example Code

```python
from equities_toolkit import Stock, Stocks_Composite, Investor, Adaptive_Price_Drop

# Initialize stock data
stock_symbols = ['AAPL', 'MSFT', 'GOOGL']
composite = Stocks_Composite(stock_symbols)
investor = Adaptive_Price_Drop(composite, threshold=5.0, cash=10000, cashflow=500, freq='W')

# Run investment simulation
start_date = '2022-01-01'
end_date = '2024-01-01'
for date in pd.date_range(start=start_date, end=end_date, freq='W'):
    investor.apd_invest(date)

# Output performance metrics
investor.calculate_total_invested()
print("Total Invested:", investor.total_invested)
```

## How is works

Overview:

1. Data Preparation: Pull historical stock and CPI data, adjusting prices for inflation.

2. Investment Strategy:
   DCA: Allocates a fixed amount periodically.
   APD: Reweights based on price change percentiles.

3. Analysis: Returns are annualized, and gain/loss metrics are calculated. Kernel density
   estimations (KDE) model price changes, with bandwidth optimized via grid search.

The notebook: dollar_cost_averaging_with_adaptive_price_drop.ipynb walks through the code and analysis.
See the notebook for more detail.

## Code Structure

equities_toolkit.py: Contains core classes and functions:

	Stock: Handles individual stock data and inflation adjustments.
	
	Stocks_Composite: Creates an index of stocks, allowing weighted portfolio analysis.
	
	Investor: Simulates investment strategies.
	
	Adaptive_Price_Drop: Implements the APD investment strategy.
	
data/: (Optional) Directory to store historical CPI data or other relevant datasets.

## Future Work

   Add CAPM analysis, threshold for large price changes above 95th percentile.

## License

This project is licensed under the MIT License - see the LICENSE file for details.