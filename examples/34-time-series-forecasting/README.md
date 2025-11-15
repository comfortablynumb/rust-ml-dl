# Time Series Forecasting

Predicting future values based on historical patterns in sequential data.

## Overview

Time series forecasting is critical for business planning, finance, operations, and scientific modeling. Unlike regular ML where data points are independent, time series data has inherent temporal ordering.

## Running

```bash
cargo run --package time-series-forecasting
```

## Time Series Components

```
Y(t) = Trend + Seasonality + Cycles + Noise

• Trend: Long-term increase/decrease
• Seasonality: Regular periodic patterns
• Cycles: Longer-term oscillations
• Noise: Random variations
```

### Decomposition Types

**Additive**: `Y(t) = T(t) + S(t) + R(t)`
- Use when seasonal variation is constant

**Multiplicative**: `Y(t) = T(t) × S(t) × R(t)`
- Use when seasonal variation increases with level

## Classical Methods

### 1. Moving Average (MA)

```
Smooth data by averaging over window

MA(k) = (y_t + y_{t-1} + ... + y_{t-k+1}) / k

✅ Simple, removes noise
❌ Lags behind actual data
```

### 2. Exponential Smoothing

```
Weighted average, more weight to recent

Simple: ŷ_{t+1} = α·y_t + (1-α)·ŷ_t

Double: Accounts for trend
Triple (Holt-Winters): Trend + seasonality
```

### 3. ARIMA (AutoRegressive Integrated Moving Average)

**Most popular classical method**

```
ARIMA(p, d, q)
  p: AutoRegressive terms (use past values)
  d: Differencing (make stationary)
  q: Moving Average terms (use past errors)
```

**Components**:

**AR (AutoRegressive)**:
```
y_t = c + φ_1·y_{t-1} + φ_2·y_{t-2} + ... + ε_t

Current value depends on past values
Like linear regression on lagged values
```

**I (Integrated - Differencing)**:
```
Make non-stationary → stationary

First difference: Δy_t = y_t - y_{t-1}
Second difference: Δ²y_t = Δy_t - Δy_{t-1}
```

**MA (Moving Average)**:
```
y_t = μ + ε_t + θ_1·ε_{t-1} + θ_2·ε_{t-2} + ...

Current value depends on past forecast errors
```

**Stationarity required**:
- Constant mean over time
- Constant variance over time
- No seasonality

### SARIMA (Seasonal ARIMA)

```
SARIMA(p,d,q)(P,D,Q)_s

Lowercase: Non-seasonal
Uppercase: Seasonal
s: Period (12 for monthly, 7 for daily/weekly)
```

## Prophet (Facebook)

Modern approach with additive components

```
y(t) = g(t) + s(t) + h(t) + ε_t

g(t): Trend (piecewise linear or logistic)
s(t): Seasonality (Fourier series)
h(t): Holidays and events
ε_t: Error term
```

### Advantages over ARIMA

✅ Handles missing data
✅ Automatic changepoint detection
✅ Multiple seasonalities (daily, weekly, yearly)
✅ Holiday effects
✅ Works on non-stationary data
✅ Robust to outliers
✅ Interpretable parameters

### Use Cases

- Business metrics (sales, users, revenue)
- Strong seasonal patterns
- Data with missing values
- Need to model holidays/events

## LSTM for Time Series

Deep learning approach using Long Short-Term Memory

### Architecture

```
Input: [x_{t-n}, ..., x_{t-1}]  (window)
  ↓
LSTM layers (capture temporal patterns)
  ↓
Dense layer
  ↓
Output: ŷ_t  (prediction)
```

### Sequence-to-Sequence (Multi-step)

```
Encoder LSTM → Context → Decoder LSTM
  ↓
[ŷ_t, ŷ_{t+1}, ..., ŷ_{t+h}]  (h-step forecast)
```

### When to Use

**LSTM**:
✅ Large dataset (10K+ points)
✅ Complex non-linear patterns
✅ Multivariate forecasting
✅ Have GPU for training

**ARIMA/Prophet**:
✅ Small dataset (100-1000 points)
✅ Need interpretability
✅ Simple patterns
✅ Fast deployment

## Multivariate Forecasting

Forecast multiple related series simultaneously

### Example: Store Sales

```
Variables:
  • Sales (target)
  • Temperature
  • Promotions
  • Foot traffic

Each influences the others!
```

### Approaches

**1. VAR (Vector AutoRegression)**
```
Extension of AR to multiple series
Each series regressed on lagged values of all series
```

**2. Multivariate LSTM**
```
Input: Multiple time series
Output: Forecast for all series
Learns cross-series dependencies automatically
```

**3. Attention-based Encoder-Decoder**
```
Attention focuses on relevant variables/timesteps
State-of-the-art for complex multivariate
```

## Feature Engineering

### 1. Lag Features
```
lag_1 = y_{t-1}
lag_2 = y_{t-2}
lag_7 = y_{t-7}  (last week)
```

### 2. Rolling Statistics
```
rolling_mean_7 = mean of last 7 days
rolling_std_7 = std of last 7 days
rolling_min_7, rolling_max_7
```

### 3. Date Features
```
day_of_week (0-6)
month (1-12)
quarter (1-4)
is_weekend (binary)
is_holiday (binary)
```

### 4. Cyclical Encoding
```
month_sin = sin(2π·month/12)
month_cos = cos(2π·month/12)

Captures cyclical nature better than raw numbers
```

### 5. Differencing
```
diff_1 = y_t - y_{t-1}  (first difference)
pct_change = (y_t - y_{t-1}) / y_{t-1}  (% change)
```

## Evaluation Metrics

### MAE (Mean Absolute Error)
```
MAE = (1/n) Σ |y_t - ŷ_t|

✅ Easy to interpret (same units as data)
✅ Robust to outliers
```

### RMSE (Root Mean Squared Error)
```
RMSE = √[(1/n) Σ (y_t - ŷ_t)²]

✅ Penalizes large errors more
❌ Sensitive to outliers
```

### MAPE (Mean Absolute Percentage Error)
```
MAPE = (100/n) Σ |y_t - ŷ_t| / y_t

✅ Scale-independent
❌ Undefined for y_t = 0
❌ Asymmetric
```

### SMAPE (Symmetric MAPE)
```
SMAPE = (100/n) Σ |y_t - ŷ_t| / (|y_t| + |ŷ_t|)

✅ Symmetric, bounded 0-100%
✅ Handles zeros better
```

## Cross-Validation

**Cannot use random splitting!** (would leak future into past)

### Time Series Split (Walk-forward)

```
Fold 1: Train [1:100]  → Test [101:110]
Fold 2: Train [1:110]  → Test [111:120]
Fold 3: Train [1:120]  → Test [121:130]
...

Always train on past, test on future!
```

## Applications

### Finance
- Stock price prediction
- Portfolio optimization
- Risk management
- Algorithmic trading

### Retail
- Demand forecasting
- Inventory optimization
- Price optimization
- Sales planning

### Energy
- Electricity demand
- Renewable energy prediction (solar, wind)
- Grid management

### Weather
- Temperature forecasting
- Precipitation prediction
- Extreme event prediction

### Healthcare
- Disease outbreak prediction
- Patient admission forecasting
- Medical resource planning

### Operations
- Website traffic
- Server load prediction
- Capacity planning

## Practical Tips

### 1. Always Visualize First!
```
Identify:
  • Trends
  • Seasonality
  • Outliers
  • Changepoints
```

### 2. Check Stationarity
```
ADF test (Augmented Dickey-Fuller)

If non-stationary:
  • Difference the series
  • Remove trend
  • Log transform (for multiplicative)
```

### 3. Start Simple
```
1. Baseline: Naïve (use last value)
2. Moving average
3. ARIMA/Prophet
4. LSTM (if have data + complexity)
```

### 4. Handle Outliers
```
Can significantly affect forecasts

Solutions:
  • Remove (if data error)
  • Cap (winsorize)
  • Model explicitly (with indicators)
```

### 5. Ensemble Models
```
Combine multiple approaches
Often better than single model

Example:
  Final = 0.5×ARIMA + 0.3×LSTM + 0.2×Prophet
```

### 6. Monitor and Retrain
```
Patterns change over time (concept drift)
Retrain periodically with recent data

Monitoring:
  • Track forecast errors
  • Alert if degradation
  • Automatic retraining pipeline
```

## Common Pitfalls

### ❌ Overfitting
```
Too many parameters → memorizes noise

Solutions:
  • Use validation set
  • Regularization
  • Simpler model
```

### ❌ Data Leakage
```
Using future information for prediction

Be careful with:
  • Feature engineering
  • Rolling statistics (use expanding, not all data)
  • Normalization (fit on train only)
```

### ❌ Ignoring Seasonality
```
Can cause large systematic errors

Solutions:
  • Use seasonal models (SARIMA, Prophet)
  • Add seasonal features
  • Seasonal differencing
```

### ❌ Wrong Evaluation
```
Using random split instead of time-based

Always:
  • Test on future data only
  • Use time series cross-validation
  • Report metrics on multiple horizons
```

## Papers & Resources

- [ARIMA Models for Time Series Forecasting](https://otexts.com/fpp3/arima.html) (Hyndman & Athanasopoulos)
- [Prophet: Forecasting at Scale](https://peerj.com/preprints/3190/) (Taylor & Letham, 2017)
- [Deep Learning for Time Series](https://arxiv.org/abs/2004.13408) (Lim & Zohren, 2021)
- [N-BEATS](https://arxiv.org/abs/1905.10437) (Oreshkin et al., 2019) - Neural basis expansion

## Key Takeaways

✓ Time series: Sequential data where order matters
✓ Components: Trend, seasonality, cycles, noise
✓ Classical: ARIMA (statistical), Prophet (additive)
✓ Deep learning: LSTM for complex patterns, large data
✓ Feature engineering: Lags, rolling stats, date features
✓ Evaluation: Time-based splits, not random
✓ Applications: Finance, retail, energy, weather, operations
✓ Start simple, add complexity as needed
✓ Monitor and retrain (patterns change over time)
