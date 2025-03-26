import equities_toolkit as etk
import pandas as pd
import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neighbors import KernelDensity
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

"""Test cases, scratch code for the equities_toolkit"""

"""SCRATCH: working toward getting weekly stock prices for the last 4 years
adjusted for inflation."""

end = datetime.today().date()
start = end + relativedelta(weeks=-208)  # 4 years ago


# Selected example stocks
tickers = ['MSFT', 'ABBV', 'XOM', 'T', 'MMM']
sc = Stocks_Composite(tickers, start=start, end=end)
eg_date = sc.stocks['MSFT'].price_history.Date[104]  # 2 years ago date

# Example index investor
II = Index_Fund(sc, cash=1000.0, cashflow=300.0)
II.set_cashflow_start(II_cf_start)
II.invest(eg_date)
II.invest(II_date_wk2)
II.invest(II_date_wk3)

# Example adaptive price drop investor
APD = Adaptive_Price_Drop(sc, cash=1000.0, cashflow=300.0)
APD.set_cashflow_start(II_cf_start)
APD.apd_invest(eg_date)
APD.apd_invest(II_date_wk2)
APD.apd_invest(II_date_wk3)

"""END SCRATCH"""


"""Quick'n'Dirty"""
import equities_toolkit as etk
import datetime

tickers = ['MSFT', 'ABBV', 'XOM', 'T', 'MMM']
end = datetime.datetime(2024, 12, 31)
start = datetime.datetime(2018, 1, 1)
bc5 = etk.Stocks_Composite(tickers, start=start, end=end, interval='1wk')
bc5.write_to_csv('bc5')
dates = pd.to_datetime(bc5.index.Date)
start_ndx = dates[dates.dt.year == 2020].index[0]
end_ndx = dates.index[-1]
end_date = dates.iloc[-1].date()
start_date = dates.iloc[start_ndx].date()
simulated_dates = bc5.index.Date.iloc[start_ndx:]
date_range_str = '20180101-20241231'
f_prefix = 'bc5_' + date_range_str
final_prices = bc5.get_prices(end_date)


"""Create strategies and run simulation"""
apd_bc5_index = etk.Adaptive_Price_Drop(bc5, threshold=5.0, cash=500.0, cashflow=500.0)
apd_bc5_index.set_cashflow_start(simulated_dates.iloc[0])
apd_bc5_index.partial_shares
apd_bc5_index.set_window(100, window_to_file=True)
apd_bc5_index.window

dca_bc5_index = etk.Index_Fund(bc5, cash=500.0, cashflow=500.0)
dca_bc5_index.set_cashflow_start(simulated_dates.iloc[0])
dca_bc5_index.partial_shares

# Apply strategies to bc5 index over simulated_dates
for d in simulated_dates:
    dca_bc5_index.invest(d)
    apd_bc5_index.apd_invest(d)

# Save simulation logs
apd_last_df = pd.DataFrame(apd_bc5_index.apd_lasts_history)
apd_weights_df = pd.DataFrame(apd_bc5_index.apd_weights_history)
apd_percentile_df = pd.DataFrame(apd_bc5_index.apd_percentile_history)
apd_last_df.to_csv('apd_lasts.log', sep=' ', index=False)
apd_weights_df.to_csv('apd_weights.log', sep=' ', index=False)
apd_percentile_df.to_csv('apd_percentile.log', sep=' ', index=False)

"""Calculate average purchase price and standard deviation"""
# DCA
dca_bc5_purchases_df = pd.DataFrame(dca_bc5_index.purchases)
dca_bc5_purchases_df_clean = dca_bc5_purchases_df.dropna()
dca_bc5_purchases_df_clean = etk.split_shares_prices(dca_bc5_purchases_df_clean, tickers)
dca_bc5_purchases_df_clean.to_csv('DCA_bc5_purchases.csv', index_label='week', sep=' ')

dca_bc5_avg_std = {}

dca_bc5_avg_std['MSFT_avg'], dca_bc5_avg_std['MSFT_std'] = etk.weighted_avg_and_std(dca_bc5_purchases_df_clean.MSFT_price, dca_bc5_purchases_df_clean.MSFT_shares)
dca_bc5_avg_std['ABBV_avg'], dca_bc5_avg_std['ABBV_std'] = etk.weighted_avg_and_std(dca_bc5_purchases_df_clean.ABBV_price, dca_bc5_purchases_df_clean.ABBV_shares)
dca_bc5_avg_std['XOM_avg'], dca_bc5_avg_std['XOM_std'] = etk.weighted_avg_and_std(dca_bc5_purchases_df_clean.XOM_price, dca_bc5_purchases_df_clean.XOM_shares)
dca_bc5_avg_std['T_avg'], dca_bc5_avg_std['T_std'] = etk.weighted_avg_and_std(dca_bc5_purchases_df_clean.T_price, dca_bc5_purchases_df_clean.T_shares)
dca_bc5_avg_std['MMM_avg'], dca_bc5_avg_std['MMM_std'] = etk.weighted_avg_and_std(dca_bc5_purchases_df_clean.MMM_price, dca_bc5_purchases_df_clean.MMM_shares)

# APD
apd_bc5_purchases_df = pd.DataFrame(apd_bc5_index.purchases)
apd_bc5_purchases_df_clean = apd_bc5_purchases_df.dropna()
apd_bc5_purchases_df_clean = etk.split_shares_prices(apd_bc5_purchases_df_clean, tickers)
apd_bc5_purchases_df_clean.to_csv('APD_bc5_purchases.csv', index_label='week', sep=' ')

apd_bc5_avg_std = {}

apd_bc5_avg_std['MSFT_avg'], apd_bc5_avg_std['MSFT_std'] = etk.weighted_avg_and_std(apd_bc5_purchases_df_clean.MSFT_price, apd_bc5_purchases_df_clean.MSFT_shares)
apd_bc5_avg_std['ABBV_avg'], apd_bc5_avg_std['ABBV_std'] = etk.weighted_avg_and_std(apd_bc5_purchases_df_clean.ABBV_price, apd_bc5_purchases_df_clean.ABBV_shares)
apd_bc5_avg_std['XOM_avg'], apd_bc5_avg_std['XOM_std'] = etk.weighted_avg_and_std(apd_bc5_purchases_df_clean.XOM_price, apd_bc5_purchases_df_clean.XOM_shares)
apd_bc5_avg_std['T_avg'], apd_bc5_avg_std['T_std'] = etk.weighted_avg_and_std(apd_bc5_purchases_df_clean.T_price, apd_bc5_purchases_df_clean.T_shares)
apd_bc5_avg_std['MMM_avg'], apd_bc5_avg_std['MMM_std'] = etk.weighted_avg_and_std(apd_bc5_purchases_df_clean.MMM_price, apd_bc5_purchases_df_clean.MMM_shares)

# Market
market_avg_std = {}
for k, v in bc5.stocks.items():
    key = k+'_avg'
    market_avg_std[key] = np.average(v.price_history.Close[start_ndx:])
    key = k+'_std'
    market_avg_std[key] = np.std(v.price_history.Close[start_ndx:])

avg_std = pd.DataFrame([dca_bc5_avg_std, apd_bc5_avg_std, market_avg_std], index=['dca', 'apd', 'market'])
avg_std
avg_std.to_csv('DCA-APD_bc5_20180101-20241231_avg_std.csv', index_label='strategy', sep=' ')



"""Calculate total gain/loss"""
dca_bc5_purchases_tuple_df = pd.DataFrame(dca_bc5_index.purchases)
dca_bc5_purchases_tuple_df_clean = dca_bc5_purchases_df.dropna()
dca_bc5_purchases_tuple_df_clean.head()

dca_bc5_cost_gain = etk.calc_cost_gains(dca_bc5_purchases_tuple_df_clean, tickers, final_prices)
dca_bc5_gl = dca_bc5_cost_gain.apply(np.sum, axis=0).to_dict()
dca_bc5_gl

apd_bc5_purchases_tuple_df = pd.DataFrame(apd_bc5_index.purchases)
apd_bc5_purchases_tuple_df_clean = apd_bc5_purchases_df.dropna()
apd_bc5_purchases_tuple_df_clean.head()

apd_bc5_cost_gain = etk.calc_cost_gains(apd_bc5_purchases_tuple_df_clean, tickers, final_prices)
apd_bc5_gl = apd_bc5_cost_gain.apply(np.sum, axis=0).to_dict()
apd_bc5_gl

gl = pd.DataFrame([dca_bc5_gl, apd_bc5_gl], index=['dca', 'apd'])
gl.to_csv('DCA-APD_bc5_20180101-20241231_gl.txt', index_label='strategy', sep=' ')



"""APD e.g. 2022-08-29"""

tw_df = pd.read_csv('./apd_window_20220829_ndx_243.csv', sep=' ')
tw_df.set_index('week')
Date = datetime.datetime.strptime('2022-08-29', '%Y-%m-%d').date()
apd_window_probs = pd.DataFrame()

threshold = 5
tw_5th_ntile_scores = {k: scoreatpercentile(tw_df[k], threshold) for k in tw_df.columns}
tw_5th_ntile_probs = {}

# Create KDE for tw
bw_bounds = (-2, 1)
params = {'bandwidth': np.logspace(bw_bounds[0], bw_bounds[1], 15)}

grid = GridSearchCV(
                KernelDensity(kernel='gaussian'),
                params,
                cv=ShuffleSplit(n_splits=5, test_size=0.20),
                n_jobs=-1
                )

# MSFT
X_MSFT = pd.DataFrame(tw_df.MSFT)

grid.fit(X_MSFT)
sample_range = pd.DataFrame(np.linspace(X_MSFT.min()*1.5, X_MSFT.max()*1.5, num=200))
apd_window_probs['MSFT'] = sample_range
apd_window_probs['MSFT_prob'] = np.exp(grid.score_samples(sample_range))
tw_5th_ntile_probs['MSFT'] = np.exp(grid.score(pd.DataFrame([tw_5th_ntile_scores['MSFT']])))
del grid, sample_range

# ABBV
X_ABBV = pd.DataFrame(tw_df.ABBV)

grid.fit(X_ABBV)
sample_range = pd.DataFrame(np.linspace(X_ABBV.min()*1.5, X_ABBV.max()*1.5, num=200))
apd_window_probs['ABBV'] = sample_range
apd_window_probs['ABBV_prob'] = np.exp(grid.score_samples(sample_range))
tw_5th_ntile_probs['ABBV'] = np.exp(grid.score(pd.DataFrame([tw_5th_ntile_scores['ABBV']])))
del grid, sample_range

# XOM
X_XOM = pd.DataFrame(tw_df.XOM)

grid.fit(X_XOM)
sample_range = pd.DataFrame(np.linspace(X_XOM.min()*1.5, X_XOM.max()*1.5, num=200))
apd_window_probs['XOM'] = sample_range
apd_window_probs['XOM_prob'] = np.exp(grid.score_samples(sample_range))
tw_5th_ntile_probs['XOM'] = np.exp(grid.score(pd.DataFrame([tw_5th_ntile_scores['XOM']])))
del grid, sample_range

# T
X_T = pd.DataFrame(tw_df['T'])

grid.fit(X_T)
sample_range = pd.DataFrame(np.linspace(X_T.min()*1.5, X_T.max()*1.5, num=200))
apd_window_probs['T'] = sample_range
apd_window_probs['T_prob'] = np.exp(grid.score_samples(sample_range))
tw_5th_ntile_probs['T'] = np.exp(grid.score(pd.DataFrame([tw_5th_ntile_scores['T']])))
del grid, sample_range

# MMM
X_MMM = pd.DataFrame(tw_df.MMM)

grid.fit(X_MMM)
sample_range = pd.DataFrame(np.linspace(X_MMM.min()*1.5, X_MMM.max()*1.5, num=200))
apd_window_probs['MMM'] = sample_range
apd_window_probs['MMM_prob'] = np.exp(grid.score_samples(sample_range))
tw_5th_ntile_probs['MMM'] = np.exp(grid.score(pd.DataFrame([tw_5th_ntile_scores['MMM']])))


