import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

"""Test cases, scratch code for the equities_toolkit"""

"""SCRATCH: working toward getting weekly stock prices for the last 4 years
adjusted for inflation."""

end = datetime.today().date()
start = end + relativedelta(weeks=-208)  # 4 years ago
eg_date = sc.stocks['MSFT'].price_history.Date[104]  # 2 years ago date

# Selected example stocks
tickers = ['MSFT', 'ABBV', 'XOM', 'T', 'MMM']
sc = Stocks_Composite(tickers, start=start, end=end)

# Example index investor
II = Index_Fund(sc, cash=1000.0, cashflow=300.0)
II.set_cashflow_start(II_cf_start)
II.invest(eg_date, partial_shares=True)
II.invest(II_date_wk2, partial_shares=True)
II.invest(II_date_wk3, partial_shares=True)

# Example adaptive price drop investor
APD = Adaptive_Price_Drop(sc, cash=1000.0, cashflow=300.0)
APD.set_cashflow_start(II_cf_start)
APD.set_partial_shares(True)
APD.apd_invest(eg_date)
APD.apd_invest(II_date_wk2)
APD.apd_invest(II_date_wk3)

"""END SCRATCH"""
