// ============================================================================
// Time Series Forecasting
// ============================================================================
//
// Predicting future values based on historical patterns in sequential data.
// Critical for business planning, finance, operations, and scientific modeling.
//
// WHAT IS TIME SERIES DATA?
// --------------------------
//
// Time series: Sequence of observations indexed by time
//   • Stock prices: Daily closing prices over years
//   • Weather: Temperature recorded every hour
//   • Sales: Monthly revenue for past 5 years
//   • Website traffic: Hourly visitors
//
// Key characteristic: Order matters! (Unlike regular ML where rows are independent)
//
// COMPONENTS OF TIME SERIES:
// ---------------------------
//
// 1. Trend (T):
//    • Long-term increase or decrease
//    • Example: Population growth, company revenue over years
//
// 2. Seasonality (S):
//    • Regular periodic fluctuations
//    • Example: Ice cream sales (high in summer, low in winter)
//    • Can be daily, weekly, monthly, yearly
//
// 3. Cyclical (C):
//    • Longer-term oscillations (not fixed period)
//    • Example: Economic cycles, business cycles
//
// 4. Residual/Noise (R):
//    • Random variations
//    • What's left after removing trend, seasonality, cycles
//
// Decomposition:
//   Additive:  Y(t) = T(t) + S(t) + R(t)
//   Multiplicative: Y(t) = T(t) × S(t) × R(t)
//
// CLASSICAL METHODS:
// ------------------
//
// 1. MOVING AVERAGE (MA):
//    • Smooth data by averaging over window
//    • MA(k) = (y_t + y_{t-1} + ... + y_{t-k+1}) / k
//    • Simple but effective for removing noise
//
// 2. EXPONENTIAL SMOOTHING:
//    • Weighted average giving more weight to recent observations
//    • Simple: ŷ_{t+1} = α·y_t + (1-α)·ŷ_t
//    • Double: Accounts for trend
//    • Triple (Holt-Winters): Accounts for trend + seasonality
//
// 3. ARIMA (AutoRegressive Integrated Moving Average):
//    • Most popular classical method
//    • ARIMA(p, d, q)
//      - p: AutoRegressive terms (use past values)
//      - d: Differencing (make series stationary)
//      - q: Moving Average terms (use past errors)
//
// ARIMA MODEL:
// ------------
//
// Components:
//
// 1. AR (AutoRegressive):
//    y_t = c + φ_1·y_{t-1} + φ_2·y_{t-2} + ... + φ_p·y_{t-p} + ε_t
//    • Current value depends on past values
//    • Like linear regression on lagged values
//
// 2. I (Integrated - Differencing):
//    • Make non-stationary series stationary
//    • First difference: Δy_t = y_t - y_{t-1}
//    • Second difference: Δ²y_t = Δy_t - Δy_{t-1}
//
// 3. MA (Moving Average):
//    y_t = μ + ε_t + θ_1·ε_{t-1} + θ_2·ε_{t-2} + ... + θ_q·ε_{t-q}
//    • Current value depends on past forecast errors
//
// Full ARIMA(p,d,q):
//   1. Difference d times to make stationary
//   2. Apply AR(p) + MA(q) to differenced series
//
// Stationarity:
//   • Mean constant over time
//   • Variance constant over time
//   • No seasonality
//   • Required for ARIMA
//
// Example ARIMA(1,1,1):
//   First difference: Δy_t = y_t - y_{t-1}
//   AR(1): Δy_t = φ_1·Δy_{t-1} + ε_t + θ_1·ε_{t-1}
//
// SEASONAL ARIMA (SARIMA):
// -------------------------
//
// SARIMA(p,d,q)(P,D,Q)_s
//   • Lowercase: Non-seasonal components
//   • Uppercase: Seasonal components
//   • s: Seasonal period (12 for monthly data, 7 for daily with weekly pattern)
//
// Example SARIMA(1,1,1)(1,1,1)_12 for monthly data:
//   • Regular ARIMA(1,1,1) for short-term patterns
//   • Seasonal ARIMA(1,1,1) for yearly patterns
//
// PROPHET (Facebook):
// -------------------
//
// Additive model: y(t) = g(t) + s(t) + h(t) + ε_t
//
// Components:
// 1. g(t): Trend
//    • Piecewise linear or logistic growth
//    • Automatically detects changepoints
//
// 2. s(t): Seasonality
//    • Fourier series for flexible seasonal patterns
//    • Can model multiple seasonalities (daily, weekly, yearly)
//
// 3. h(t): Holidays and events
//    • Special dates with custom effects
//    • Example: Black Friday, Christmas
//
// 4. ε_t: Error term
//
// Advantages over ARIMA:
//   ✅ Handles missing data
//   ✅ Automatic changepoint detection
//   ✅ Multiple seasonalities
//   ✅ Holiday effects
//   ✅ Works on non-stationary data
//   ✅ Robust to outliers
//   ✅ Interpretable parameters
//
// Use cases:
//   • Business metrics (sales, users, revenue)
//   • Data with strong seasonal patterns
//   • Data with missing values
//   • Need to model holidays/events
//
// LSTM FOR TIME SERIES:
// ---------------------
//
// Deep learning approach using LSTM (Long Short-Term Memory)
//
// Why LSTM for time series?
//   • Learns complex patterns automatically
//   • Handles long-term dependencies
//   • Multivariate (multiple related series)
//   • Non-linear relationships
//
// Architecture:
// ```
// Input: [x_{t-n}, x_{t-n+1}, ..., x_{t-1}]  (window of past values)
//   ↓
// LSTM layers (capture temporal patterns)
//   ↓
// Dense layer
//   ↓
// Output: ŷ_t  (prediction for time t)
// ```
//
// Sequence-to-Sequence for multi-step:
// ```
// Encoder LSTM: Process input sequence
//   ↓
// Decoder LSTM: Generate output sequence
//   ↓
// Output: [ŷ_t, ŷ_{t+1}, ..., ŷ_{t+h}]  (h-step ahead forecast)
// ```
//
// Advantages:
//   ✅ Automatic feature learning
//   ✅ Handles multivariate time series
//   ✅ Captures non-linear patterns
//   ✅ Can model complex dependencies
//
// Disadvantages:
//   ❌ Needs lots of data (1000s of points)
//   ❌ Harder to interpret
//   ❌ Computationally expensive
//   ❌ Sensitive to hyperparameters
//
// When to use LSTM:
//   • Large dataset (10K+ points)
//   • Complex patterns (non-linear, multiple seasonalities)
//   • Multivariate forecasting
//   • Have GPU for training
//
// When to use ARIMA/Prophet:
//   • Small dataset (100-1000 points)
//   • Need interpretability
//   • Simple patterns
//   • Fast deployment
//
// MULTIVARIATE FORECASTING:
// --------------------------
//
// Forecast multiple related time series simultaneously
//
// Example: Forecasting store sales
//   • Variables: Sales, temperature, promotions, foot traffic
//   • Each influences the others
//
// Approaches:
//
// 1. VAR (Vector AutoRegression):
//    • Extension of AR to multiple series
//    • Each series regressed on lagged values of all series
//
//    y1_t = c1 + Σ φ1j·y1_{t-j} + Σ φ12j·y2_{t-j} + ε1_t
//    y2_t = c2 + Σ φ21j·y1_{t-j} + Σ φ22j·y2_{t-j} + ε2_t
//
// 2. Multivariate LSTM:
//    • Input: Multiple time series
//    • Output: Forecast for all series
//    • Learns cross-series dependencies automatically
//
// 3. Encoder-Decoder with attention:
//    • Attention mechanism focuses on relevant variables/timesteps
//    • State-of-the-art for complex multivariate forecasting
//
// FEATURE ENGINEERING FOR TIME SERIES:
// -------------------------------------
//
// 1. Lag features:
//    • Use past values as features
//    • lag_1 = y_{t-1}, lag_2 = y_{t-2}, ...
//
// 2. Rolling statistics:
//    • rolling_mean_7 = mean of last 7 days
//    • rolling_std_7 = std of last 7 days
//    • rolling_min_7, rolling_max_7
//
// 3. Date features:
//    • day_of_week (0-6)
//    • month (1-12)
//    • quarter (1-4)
//    • is_weekend (binary)
//    • is_holiday (binary)
//
// 4. Cyclical encoding:
//    • month_sin = sin(2π·month/12)
//    • month_cos = cos(2π·month/12)
//    • Captures cyclical nature
//
// 5. Differencing:
//    • diff_1 = y_t - y_{t-1} (first difference)
//    • pct_change = (y_t - y_{t-1}) / y_{t-1}
//
// EVALUATION METRICS:
// -------------------
//
// 1. MAE (Mean Absolute Error):
//    MAE = (1/n) Σ |y_t - ŷ_t|
//    • Easy to interpret (same units as data)
//
// 2. RMSE (Root Mean Squared Error):
//    RMSE = √[(1/n) Σ (y_t - ŷ_t)²]
//    • Penalizes large errors more
//
// 3. MAPE (Mean Absolute Percentage Error):
//    MAPE = (100/n) Σ |y_t - ŷ_t| / y_t
//    • Scale-independent (useful for comparing different series)
//
// 4. SMAPE (Symmetric MAPE):
//    SMAPE = (100/n) Σ |y_t - ŷ_t| / (|y_t| + |ŷ_t|)
//    • Symmetric, bounded 0-100%
//
// CROSS-VALIDATION FOR TIME SERIES:
// ----------------------------------
//
// Cannot use random splitting! (would leak future into past)
//
// Time Series Split (Walk-forward validation):
// ```
// Fold 1: Train [1:100] → Test [101:110]
// Fold 2: Train [1:110] → Test [111:120]
// Fold 3: Train [1:120] → Test [121:130]
// ...
// ```
//
// Always train on past, test on future!
//
// APPLICATIONS:
// -------------
//
// 1. Finance:
//    • Stock price prediction
//    • Portfolio optimization
//    • Risk management
//    • Algorithmic trading
//
// 2. Retail:
//    • Demand forecasting
//    • Inventory optimization
//    • Price optimization
//    • Sales planning
//
// 3. Energy:
//    • Electricity demand forecasting
//    • Renewable energy prediction (solar, wind)
//    • Grid management
//
// 4. Weather:
//    • Temperature forecasting
//    • Precipitation prediction
//    • Extreme event prediction
//
// 5. Healthcare:
//    • Disease outbreak prediction
//    • Patient admission forecasting
//    • Medical resource planning
//
// 6. Operations:
//    • Website traffic forecasting
//    • Server load prediction
//    • Capacity planning
//
// PRACTICAL TIPS:
// ---------------
//
// 1. Always visualize the data first!
//    • Identify trends, seasonality, outliers
//    • Choose appropriate method
//
// 2. Check for stationarity:
//    • ADF test (Augmented Dickey-Fuller)
//    • If non-stationary, difference or detrend
//
// 3. Start simple:
//    • Try moving average baseline
//    • Then ARIMA/Prophet
//    • LSTM if you have enough data and complexity
//
// 4. Handle outliers:
//    • Can significantly affect forecasts
//    • Remove, cap, or model explicitly
//
// 5. Multiple models:
//    • Ensemble different approaches
//    • Often better than single model
//
// 6. Monitor and retrain:
//    • Patterns change over time (concept drift)
//    • Retrain periodically with recent data
//
// COMMON PITFALLS:
// ----------------
//
// ❌ Overfitting:
//    • Too many parameters, memorizes noise
//    • Use validation set, regularization
//
// ❌ Data leakage:
//    • Using future information for prediction
//    • Careful with feature engineering
//
// ❌ Ignoring seasonality:
//    • Can cause large errors
//    • Use seasonal models (SARIMA, Prophet)
//
// ❌ Wrong evaluation:
//    • Using random split instead of time-based
//    • Always test on future data only
//
// KEY TAKEAWAYS:
// --------------
//
// ✓ Time series forecasting predicts future from historical patterns
// ✓ Classical: ARIMA (statistical), Prophet (additive components)
// ✓ Deep learning: LSTM for complex patterns, large data
// ✓ Components: Trend, seasonality, cycles, noise
// ✓ Feature engineering critical: lags, rolling stats, date features
// ✓ Evaluation: Use time-based splits, not random
// ✓ Applications: Finance, retail, energy, weather, healthcare
// ✓ Start simple, add complexity as needed
//
// ============================================================================

use ndarray::Array1;
use rand::Rng;

/// Simple Moving Average
fn moving_average(data: &[f32], window: usize) -> Vec<f32> {
    let mut result = Vec::new();

    for i in 0..data.len() {
        if i < window - 1 {
            result.push(data[i]); // Not enough data yet
        } else {
            let sum: f32 = data[i - window + 1..=i].iter().sum();
            result.push(sum / window as f32);
        }
    }

    result
}

/// Exponential smoothing
fn exponential_smoothing(data: &[f32], alpha: f32) -> Vec<f32> {
    let mut result = vec![data[0]]; // Initialize with first value

    for i in 1..data.len() {
        let smoothed = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        result.push(smoothed);
    }

    result
}

/// Simple seasonal decomposition (additive)
struct SeasonalDecomposition {
    trend: Vec<f32>,
    seasonal: Vec<f32>,
    residual: Vec<f32>,
}

impl SeasonalDecomposition {
    fn decompose(data: &[f32], period: usize) -> Self {
        let n = data.len();

        // 1. Calculate trend using moving average
        let mut trend = vec![0.0; n];
        for i in 0..n {
            if i < period / 2 || i >= n - period / 2 {
                trend[i] = data[i]; // Use original at boundaries
            } else {
                let start = i - period / 2;
                let end = i + period / 2;
                let sum: f32 = data[start..=end].iter().sum();
                trend[i] = sum / (period + 1) as f32;
            }
        }

        // 2. Detrend
        let detrended: Vec<f32> = data.iter()
            .zip(&trend)
            .map(|(d, t)| d - t)
            .collect();

        // 3. Calculate average seasonal pattern
        let mut seasonal_sums = vec![0.0; period];
        let mut seasonal_counts = vec![0; period];

        for i in 0..n {
            let season_idx = i % period;
            seasonal_sums[season_idx] += detrended[i];
            seasonal_counts[season_idx] += 1;
        }

        let seasonal_pattern: Vec<f32> = seasonal_sums.iter()
            .zip(&seasonal_counts)
            .map(|(sum, count)| if *count > 0 { sum / *count as f32 } else { 0.0 })
            .collect();

        // 4. Repeat seasonal pattern
        let seasonal: Vec<f32> = (0..n)
            .map(|i| seasonal_pattern[i % period])
            .collect();

        // 5. Calculate residual
        let residual: Vec<f32> = data.iter()
            .zip(&trend)
            .zip(&seasonal)
            .map(|((d, t), s)| d - t - s)
            .collect();

        Self { trend, seasonal, residual }
    }
}

/// Simple AR(1) model
struct AutoRegressive {
    phi: f32, // AR coefficient
    c: f32,   // Constant
}

impl AutoRegressive {
    fn fit(data: &[f32]) -> Self {
        // Simple least squares fit for AR(1): y_t = c + φ·y_{t-1} + ε_t
        let n = data.len() - 1;

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for i in 0..n {
            let x = data[i];
            let y = data[i + 1];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let n_f = n as f32;
        let phi = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x);
        let c = (sum_y - phi * sum_x) / n_f;

        Self { phi, c }
    }

    fn predict(&self, last_value: f32) -> f32 {
        self.c + self.phi * last_value
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Time Series Forecasting");
    println!("{}", "=".repeat(70));
    println!();

    // Generate synthetic time series with trend + seasonality + noise
    let n = 100;
    let mut rng = rand::thread_rng();

    let data: Vec<f32> = (0..n)
        .map(|t| {
            let trend = 0.5 * t as f32; // Linear trend
            let seasonal = 10.0 * (2.0 * std::f32::consts::PI * t as f32 / 12.0).sin(); // Monthly seasonality
            let noise = rng.gen_range(-2.0..2.0);
            trend + seasonal + noise + 50.0 // Base level of 50
        })
        .collect();

    println!("GENERATED TIME SERIES:");
    println!("----------------------");
    println!("Length: {} observations", n);
    println!("First 10 values: {:?}", &data[..10].iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
    println!();

    // 1. Moving Average
    println!("1. MOVING AVERAGE:");
    println!("------------------");
    let window = 7;
    let ma = moving_average(&data, window);
    println!("Window size: {}", window);
    println!("Smoothed (last 5): {:?}", &ma[ma.len()-5..].iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
    println!();

    // 2. Exponential Smoothing
    println!("2. EXPONENTIAL SMOOTHING:");
    println!("-------------------------");
    let alpha = 0.3;
    let es = exponential_smoothing(&data, alpha);
    println!("Alpha: {}", alpha);
    println!("Smoothed (last 5): {:?}", &es[es.len()-5..].iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
    println!();

    // 3. Seasonal Decomposition
    println!("3. SEASONAL DECOMPOSITION:");
    println!("--------------------------");
    let period = 12; // Monthly seasonality
    let decomp = SeasonalDecomposition::decompose(&data, period);
    println!("Period: {}", period);
    println!("Trend (last 5): {:?}", &decomp.trend[decomp.trend.len()-5..].iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
    println!("Seasonal pattern (first {}): {:?}", period, &decomp.seasonal[..period].iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
    println!();

    // 4. AutoRegressive AR(1)
    println!("4. AUTOREGRESSIVE AR(1):");
    println!("------------------------");
    let ar_model = AutoRegressive::fit(&data);
    println!("Fitted model: y_t = {:.3} + {:.3} × y_{{t-1}}", ar_model.c, ar_model.phi);

    // Forecast next 5 steps
    let mut forecast = vec![*data.last().unwrap()];
    for _ in 0..4 {
        let next = ar_model.predict(*forecast.last().unwrap());
        forecast.push(next);
    }
    println!("Last observed: {:.1}", data.last().unwrap());
    println!("5-step forecast: {:?}", &forecast[1..].iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
    println!();

    println!("{}", "=".repeat(70));
    println!("KEY CONCEPTS DEMONSTRATED:");
    println!("{}", "=".repeat(70));
    println!("✓ Moving Average: Smoothing by averaging over window");
    println!("✓ Exponential Smoothing: Weighted average (recent > old)");
    println!("✓ Seasonal Decomposition: Trend + Seasonal + Residual");
    println!("✓ AutoRegressive: Predict using past values");
    println!();
    println!("METHODS OVERVIEW:");
    println!("  • Classical: ARIMA, Prophet, Holt-Winters");
    println!("  • Deep Learning: LSTM, Seq2Seq, Temporal CNN");
    println!("  • Ensemble: Combine multiple models");
    println!();
    println!("APPLICATIONS:");
    println!("  • Finance: Stock prices, risk management");
    println!("  • Retail: Demand forecasting, inventory");
    println!("  • Energy: Load forecasting, renewable prediction");
    println!("  • Weather: Temperature, precipitation");
    println!("  • Healthcare: Disease outbreak, admissions");
    println!("  • Operations: Traffic, server load, capacity");
    println!();
    println!("BEST PRACTICES:");
    println!("  • Visualize data first (identify patterns)");
    println!("  • Check stationarity (ADF test)");
    println!("  • Use time-based cross-validation");
    println!("  • Handle outliers and missing data");
    println!("  • Monitor and retrain (concept drift)");
    println!();
}
