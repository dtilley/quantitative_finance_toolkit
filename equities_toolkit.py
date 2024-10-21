import numpy as np
import pandas as pd
import cpi
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neighbors import KernelDensity
from scipy.stats import percentileofscore


# TODO: Create inflation estimation object
def get_cpi(date):
    try:
        return cpi.get(date)
    except cpi.errors.CPIObjectDoesNotExist:
        return np.nan


def annualize_return(row, end_date: datetime, prices: dict) -> pd.Series:
    """Annualizes the return from investor purchases

    row: pandas dataframe row from investor purchase history
    end_date: final date to compare value
    prices: stock prices on end_date

    returns: the annualize percentage change
    """
    n = 52/((end_date - row.Date).days // 7)
    r_annual = {}
    r_annual['Date'] = row.Date
    for k, v in prices.items():
        s, p = row[k]  # shares, price of purchase
        r_annual[k] = ((1 + prices[k]/p) ** n) - 1
    return(pd.Series(r_annual))


def gain_loss(row, prices: dict) -> pd.Series:
    """Calculates the gain/loss of the investor purchase history.

    row: pandas dataframe row from investor purchase history
    prices: stock prices on end_date

    returns: the gain/loss of each purchase
    """
    gl = {}
    gl['Date'] = row.Date
    for k, v in prices.items():
        s, p = row[k]  # shares, price of purchase
        gl[k] = round(s*prices[k] - s*p, 2)  # value - purchase cost
    return(pd.Series(gl))


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
                cv=ShuffleSplit(n_splits=5, test_size=0.20),
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
        total_cap = sum(weights.values())
        self.weights = {k: v/total_cap for k, v in weights.items()}

        # Calculate index
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
            prices[k] = v.price_history[v.price_history.Date == date].Close.iloc[0]
        return prices

    def get_price_histroy_index(self, date: datetime) -> int:
        s = self.stocks[self.tickers[0]]
        return(int(s.price_history[s.price_history.Date == date].index.values))


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
        self.total_invested = None
        self.purchases = []  # list of dicts of purchase history

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
        date = pd.Timestamp(date)  # For period comparisons
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

    def calculate_total_invested(self):
        """Calculates the total cost of investment."""
        df = pd.DataFrame(self.purchases).dropna().drop(columns=['Date', 'Action'])
        total = df.apply(lambda row: row.apply(lambda x: x[0] * x[1]))
        self.total_invested = total.sum().sum()


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
        self.portfolio = {k: 0 for k in sc.tickers}  # initialize shares

    def invest(self, date: datetime, partial_shares=False):
        super().update_cashflow(date)  # Update avaliable cash
        prices = self.sc.get_prices(date)
        purchase = {}
        purchase['Date'] = date
        total_cost = 0.0

        for k in prices:
            if partial_shares:
                shares = (self.cash * self.weights[k]) / prices[k]
            else:
                shares = int((self.cash * self.weights[k]) / prices[k])
            cost = shares * prices[k]
            total_cost += cost
            purchase[k] = (shares, round(prices[k], 2))  # (num_shares, P/S)

        total_cost = round(total_cost, 2)  # avoids'Price Out' from rounding
        # Check if the Investor is priced out
        if self.cash >= total_cost and self.cash != 0:
            self.cash -= total_cost
            purchase['Action'] = 'Buy ' + self.descriptor
            # Successful purchase: Update portfolio
            self.portfolio = {k: v+purchase[k][0] for k, v in self.portfolio.items()}
        else:
            purchase['Action'] = 'Price Out'
            # Reduce purchase dict
            purchase = {k: v for k, v in purchase.items() if k in ('Date', 'Action')}
        self.purchases.append(purchase)


class Adaptive_Price_Drop(Investor):
    """The Adaptive_Price_Drop investor distributes stock purchases with the
    proportions of the Stocks_Composite weights by default. If one of the stocks
    in the Stocks_Composite experience a rare price drop (e.g. 5th percentile)
    then the investment is proportioned with the weight of the percentile^-1.
    The new weights priorize the stocks with the lowest price drop rather than
    strickly the market cap weights.
    """
    def __init__(self, sc: Stocks_Composite, threshold: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.sc = sc
        self.lag = None  # prior period price changes considered descriptive
        self.partial_shares = False  # whether to allow partial shares
        self.threshold = threshold  # percentile threshold
        self.default_weights = self.sc.get_weights()  # default weights
        self.default_descriptor = 'Index'
        self.portfolio = {k: 0 for k in sc.tickers}  # initialize shares

    def set_lag(self, lag: int):
        self.lag = lag

    def set_partial_shares(self, partial_shares: bool):
        self.partial_shares = partial_shares

    def get_price_change_percentile(self, ndx: int) -> dict:
        percentile = {}
        if self.lag is None or ndx-self.lag < 0:
            for k, v in self.sc.stocks.items():
                pct_change = v.price_history.Inflation_Adj[:(ndx+1)].pct_change()
                last = pct_change.pop(ndx)
                percentile[k] = percentileofscore(pct_change, last, nan_policy='omit')
        else:
            for k, v in self.sc.stocks.items:
                lwrbnd = ndx - self.lag
                pct_change = v.price_history.Inflation_Adj[lwrbnd:(ndx+1)].pct_change()
                last = pct_change.pop(ndx)
                percentile[k] = percentileofscore(pct_change, last, nan_policy='omit')
        return(percentile)

    def invest(self, date: datetime, weights: dict, descriptor: str):
        super().update_cashflow(date)  # Update avaliable cash
        purchase = {}
        purchase['Date'] = date
        prices = self.sc.get_prices(date)
        total_cost = 0.0

        for k in prices:
            if self.partial_shares:
                shares = (self.cash * weights[k]) / prices[k]
            else:
                shares = int((self.cash * weights[k]) / prices[k])
            cost = shares * prices[k]
            total_cost += cost
            purchase[k] = (shares, round(prices[k], 2))  # (num_shares, P/S)

        total_cost = round(total_cost, 2)  # avoids'Price Out' from rounding
        # Check if the Investor is priced out
        if self.cash >= total_cost and self.cash != 0:
            self.cash -= total_cost
            purchase['Action'] = 'Buy ' + descriptor
            # Successful purchase: Update portfolio
            self.portfolio = {k: v+purchase[k][0] for k, v in self.portfolio.items()}
        else:
            purchase['Action'] = 'Price Out'
            # Reduce purchase dict
            purchase = {k: v for k, v in purchase.items() if k in ('Date', 'Action')}
        self.purchases.append(purchase)

    def apd_invest(self, date: datetime):
        ndx = self.sc.get_price_histroy_index(date)

        # Check investment strategy
        if ndx < 30:
            # Not enough price history, default index weights
            self.invest(date, self.default_weights, self.default_descriptor)
        else:
            ntiles = self.get_price_change_percentile(ndx)
            if any(n <= self.threshold for n in ntiles.values()):
                # threshold met re-weighting
                n_weights = {}
                for k, v in ntiles.items():
                    try:
                        n_weights[k] = v**-1
                    except ZeroDivisionError:
                        n_weights[k] = 0.1**-1  # a low default percentile
                # Normalize weights
                total_wght = sum(n_weights.values())
                n_weights = {k: v/total_wght for k, v in n_weights.items()}
                self.invest(date, n_weights, 'APD')
            else:
                self.invest(date, self.default_weights, self.default_descriptor)
