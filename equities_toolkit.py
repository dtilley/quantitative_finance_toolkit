import numpy as np
import pandas as pd
import cpi
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
sns.set_theme()


"""SCRATCH: working toward getting weekly stock prices for the last 5 years
adjusted for inflation. Starting with MSFT"""

today = datetime.date.today()
date_100wks = today + relativedelta(weeks=-100)

# Selected example stocks
tickers = ['MSFT', 'ABBV', 'XOM', 'T', 'MMM']



# Quick Plot
plt.scatter(msft_hist.index, msft_hist.Close, marker='.')
plt.xlabel('t (wk)')
plt.ylabel('MSFT Stock Price ($)')
plt.show()

"""END SCRATCH"""


# TODO: Create inflation estimation object
def get_cpi(date):
    try:
        return cpi.get(date)
    except cpi.errors.CPIObjectDoesNotExist:
        return np.nan


class Stock:
    """Stock object that cleans up price history for analysis. The underlying
    object is the yfinance ticker.
    """
    def __init__(self, ticker: str):
        self.ticker = yf.Ticker(ticker)
        self.symbol = ticker
        self.price_history = None
        self.t_res = None
        self.price_pct_change = None
        self.kde = None

    def download_history(self, start=None, end=None, interval='1wk', close_only=True):
        """Downloads stock price history with and adjustment for inflation."""
        if start is None:
            start = datetime.date.today() - relativedelta(years=5)
        else:
            start = start
        if end is None:
            end = datetime.date.today()
        else:
            end = end
        hist = self.ticker.history(start=start, end=end, interval=interval)
        hist.reset_index(inplace=True)  # Reindex and drop time stamp
        hist.Date = pd.to_datetime(hist.Date).dt.date

        # Drop less used and zero columns
        if close_only:
            drop_columns = ['Open', 'High', 'Low'] + list(hist.columns[(hist == 0).all()])
            hist.drop(columns=drop_columns, inplace=True)

        hist['Inflation_Adj'] = hist.apply(lambda row: self.inflation_adjust(row['Close'], row['Date']), axis=1)
        self.price_history = hist
        self.t_res = interval

    def inflation_adjust(self, price, date):
        """Adjust share prices to most recent CPI"""
        try:
            return cpi.inflate(price, date)
        except cpi.errors.CPIObjectDoesNotExist:
            return price

    def build_kde(self, bw_bounds: tuple = (-2, 1)):
        """Trains a kernel density estimatior for price-change/interval."""
        if self.price_history is not None:
            # Preprocessing data + parameters
            X = pd.DataFrame(self.price_history.Inflation_Adj.pct_change() * 100)
            X.dropna(inplace=True)
            self.price_pct_change = X
            params = {'bandwidth': np.logspace(bw_bounds[0], bw_bounds[1], 15)}

            # Fit kde with bandwidth grid search
            grid = GridSearchCV(
                KernelDensity(kernel='gaussian'),
                params,
                cv=StratifiedShuffleSplit(n_splits=5, test_size=0.20),
                n_jobs=-1
                )
            grid.fit(X)

            # Check if the best_params_ is an edge case
            if grid.best_params_['bandwidth'] in params['bandwidth'][[-1, 0]]:
                bw = grid.best_params_['bandwidth']
                print(f'Warning: {round(bw, 4)} is at the edge of the bandwidth bounds.')

            self.kde = grid
        else:
            print(f"{self.symbol} price_history is 'None'. Run {self.symbol}.download_history().")


class Investor:
    """The investor object tests different investment strategies.

    Keyword arguments:
    cash   -- cash on hand
    cashflow -- cash amount that accumulates / time period
    start_date -- datetime of when cashflow payments will begin
    period -- ('W', '2W', 'M') time period cashflow disbursment
    """
    def __init__(self, cash: float = 0.0, cashflow: float = 0.0, freq='2W'):
        self.cash = cash
        self.cashflow = cashflow
        self.current_pay_period = None
        self.next_pay_period = None

        # Check cashflow frequency
        if freq in ('W', '2W', 'M'):
            self.freq = freq
        else:
            print("Pay period must be ('W', '2W', 'M').")

    def set_cashflow_start(self, start_date: datetime):
        """Sets the cashflow start date to sync with historical stock prices."""
        self.current_pay_period = pd.Period(start_date, freq=self.freq)
        self.next_pay_period = self.current_pay_period + 1

    def update_cashflow(self, date: datetime, back_pay=True):
        """Cashflow of the investor."""
        if self.current_pay_period is None:
            return  # pay does not exist / not setup

        elif self.current_pay_period.start_time <= date <= self.current_pay_period.end_time:
            pass  # a pay period has not completed yet

        elif self.next_pay_period.start_time <= date <= self.next_pay_period.end_time:
            self.cash += self.cashflow  # Payment!

            # Update pay period
            self.current_pay_period = self.next_pay_period
            self.next_pay_period += 1

        elif self.next_pay_period.end_time <= date and back_pay:
            print('Nonsequential cashflow not setup yet. Check inputs.')
            pass  # will add nonsequential later
        else:
            print('Cashflow update unsuccessful. Check inputs.')


class Index_Fund(Investor):
    """The Index_Fund investor distributes stock purchases with the proportions
    of the Stocks_Composite weights. The strategy is to broadly invest in the
    whole market defined within the Stocks_Composite object.
    """
    def __init__(self, sc: Stocks_Composite, **kwargs):
        super().__init__(**kwargs)
        self.sc = sc
        self.weights = self.sc.get_weights()
        self.descriptor = 'Index'
        self.purchases = []  # list of dicts
        self.portfolio = {}.fromkeys(cmpst_ndx.tickers)  # tickers and number of shares

    def invest(self, date: datetime, partial_shares=False):
        super().update_cashflow(date)  # Update avaliable cash
        prices = self.sc.get_prices(date)
        if partial_shares:
            
        


class Stocks_Composite:
    """Collection of stocks for analysis. The stock price history are downloaded
        a composite index created.
    """
    def __init__(self, tickers: list, **kwargs):
        self.tickers = tickers
        self.stocks = {i: Stock(i) for i in tickers}

        # Download history
        for v in self.stocks.values():
            v.download_history(**kwargs)

        # Calculate weights
        weights = {k: v.ticker.info['marketCap'] for k, v in self.stocks.items()}
        total_cap = np.sum(weights.values())
        self.weights = {k: v/total_cap for k, v in weights.items()}

        # Calcuate index
        ndx = pd.DataFrame()
        ndx_adj = pd.DataFrame()
        for k, v in self.stocks.items():
            ndx[k] = v.price_history.Close * self.weights[k]
            ndx_adj[k] = v.price_history.Inflation_Adj * self.weights[k]
        ndx = ndx.apply(np.sum, axis=1)
        ndx_adj = ndx_adj.apply(np.sum, axis=1)
        self.index = pd.concat([self.stocks[tickers[0]].price_history.Date, ndx, ndx_adj], axis=1)
        self.index.columns = ['Date', 'Index', 'Index_Adj']

    def get_weights(self) -> dict:
        return self.weights

    def get_prices(self, date: datetime) -> dict:
        prices = {}
        for k, v in self.stocks.items():
            prices[k] = v.price_history[v.price_history.Date == date].Close
        return prices
