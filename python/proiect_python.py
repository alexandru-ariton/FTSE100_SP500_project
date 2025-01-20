import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from arch import arch_model

# Load and Clean Data
def load_and_clean(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    data['Close'] = data['Close'].str.replace(',', '').astype(float)
    data.dropna(subset=['Close'], inplace=True)
    return data

# FTSE100 and S&P500 Data
ftse100 = load_and_clean('FTSE100_data.csv')
sp500 = load_and_clean('SP500_data.csv')

# Compute Daily Returns
ftse100['Return'] = np.log(ftse100['Close'] / ftse100['Close'].shift(1))
sp500['Return'] = np.log(sp500['Close'] / sp500['Close'].shift(1))

# Descriptive Statistics and Jarque-Bera Test
print("Descriptive Statistics for FTSE100 Returns:")
# Ensure 'Return' is fully clean
ftse100['Return'] = ftse100['Return'].replace([np.inf, -np.inf], np.nan).dropna()

# Drop any remaining NaN values completely
ftse100 = ftse100.dropna(subset=['Return'])

# Diagnostics
print(f"Number of valid returns after final cleaning: {len(ftse100['Return'])}")
print(f"NaN count in returns after final cleaning: {ftse100['Return'].isnull().sum()}")
print(f"Inf count in returns after final cleaning: {np.isinf(ftse100['Return']).sum()}")


descriptive_stats = ftse100['Return'].describe()
descriptive_stats['Skewness'] = skew(ftse100['Return'])
descriptive_stats['Kurtosis'] = kurtosis(ftse100['Return'])
print(descriptive_stats)

jb_stat, jb_p_value = jarque_bera(ftse100['Return'])
print(f"Jarque-Bera Test Statistic: {jb_stat}, P-Value: {jb_p_value}")

# Plot Histogram
plt.hist(ftse100['Return'], bins=50, edgecolor='black')
plt.title("Histogram of FTSE100 Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

# Beta Calculation and Jensen Test of CAPM
print("\nBeta Values and Jensen Test")
ftse100['Stock1'] = ftse100['Close'] * (1 + np.random.normal(0, 0.01, len(ftse100)))
ftse100['Stock2'] = ftse100['Close'] * (1 + np.random.normal(0, 0.01, len(ftse100)))

for stock in ['Stock1', 'Stock2']:
    ftse100[f'{stock}_Return'] = np.log(ftse100[stock] / ftse100[stock].shift(1))

X = sm.add_constant(ftse100['Return'])
for stock in ['Stock1_Return', 'Stock2_Return']:
    y = ftse100[stock].dropna()
    model = sm.OLS(y, X.loc[y.index]).fit()
    beta = model.params['Return']
    print(f"Beta for {stock}: {beta}")

# Jensen Test for Stock1
risk_free_rate = 0.01 / 252
expected_market_return = ftse100['Return'].mean()
alpha = ftse100['Stock1_Return'].mean() - (risk_free_rate + beta * (expected_market_return - risk_free_rate))
print(f"Jensen's Alpha for Stock1: {alpha}")

# SACF and SPACF
print("\nSACF and SPACF")
plot_acf(ftse100['Return'], lags=20)
plt.title("SACF for FTSE100 Returns")
plt.show()

plot_pacf(ftse100['Return'], lags=20)
plt.title("SPACF for FTSE100 Returns")
plt.show()

# Unit Root Test
print("\nUnit Root Test")

# Final cleaning for ADF Test
ftse100['Return'] = ftse100['Return'].replace([np.inf, -np.inf], np.nan).dropna()
ftse100 = ftse100.dropna(subset=['Return'])  # Drop rows with NaN in 'Return'

# Ensure clean diagnostics
print(f"Number of valid returns after final cleaning: {len(ftse100['Return'])}")
print(f"NaN count in returns after final cleaning: {ftse100['Return'].isnull().sum()}")
print(f"Inf count in returns after final cleaning: {np.isinf(ftse100['Return']).sum()}")

# Perform ADF test
if len(ftse100['Close']) > 20:
    adf_prices = adfuller(ftse100['Close'], maxlag=1)
    print(f"ADF Test for FTSE100 Prices: Statistic={adf_prices[0]}, P-Value={adf_prices[1]}")
else:
    print("Insufficient data points for ADF test on FTSE100 Prices.")

if len(ftse100['Return']) > 20:
    adf_returns = adfuller(ftse100['Return'], maxlag=1)
    print(f"ADF Test for FTSE100 Returns: Statistic={adf_returns[0]}, P-Value={adf_returns[1]}")
else:
    print("Insufficient data points for ADF test on FTSE100 Returns.")


# Co-integration Test
print("\nCo-integration Test")
ftse_close, sp500_close = ftse100['Close'].align(sp500['Close'], join='inner')
coint_stat, coint_p_value, _ = coint(ftse_close, sp500_close)
print(f"Co-integration Test Statistic: {coint_stat}, P-Value: {coint_p_value}")

# ARCH Effect and GARCH Model
print("\nARCH Effect and GARCH Model")

# Rescale returns by multiplying by 100
ftse100['Return_Rescaled'] = ftse100['Return'] * 100

# Fit ARCH model
arch_test = arch_model(ftse100['Return_Rescaled'], vol='ARCH', p=1)
arch_result = arch_test.fit(disp='off')
print("\nARCH Model Summary:")
print(arch_result.summary())

# Fit GARCH model
garch_test = arch_model(ftse100['Return_Rescaled'], vol='Garch', p=1, q=1)
garch_result = garch_test.fit(disp='off')
print("\nGARCH Model Summary:")
print(garch_result.summary())

# Plot Conditional Volatility from GARCH
plt.figure(figsize=(10, 6))
plt.plot(garch_result.conditional_volatility, label='Conditional Volatility')
plt.title("Conditional Volatility from GARCH Model (Rescaled Returns)")
plt.legend()
plt.show()

