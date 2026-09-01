import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import yfinance as yf


# ============================================================
# 1. DOWNLOAD HISTORICAL FINANCIAL DATA
# ============================================================

# Example assets
assets = {
    "Stocks": "SPY",
    "Bonds": "AGG",
    "Gold": "GLD"
}

start_date = "2015-01-01"
end_date = "2025-12-31"


print("=" * 70)
print(" VALUE-EQUIVALENCE PORTFOLIO PREDICTION")
print("=" * 70)

print("\nDownloading historical data...")


prices = pd.DataFrame()

for name, ticker in assets.items():

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    # Handle different yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]

    prices[name] = close


prices = prices.dropna()


print(
    "\nAvailable historical observations:",
    len(prices)
)


# ============================================================
# 2. CALCULATE DAILY RETURNS
# ============================================================

returns = prices.pct_change().dropna()


# ============================================================
# 3. CREATE PORTFOLIO STRATEGIES
# ============================================================

strategies = {

    "Conservative": {
        "Stocks": 0.20,
        "Bonds": 0.60,
        "Gold": 0.20
    },

    "Balanced": {
        "Stocks": 0.50,
        "Bonds": 0.30,
        "Gold": 0.20
    },

    "Growth": {
        "Stocks": 0.70,
        "Bonds": 0.20,
        "Gold": 0.10
    },

    "Aggressive": {
        "Stocks": 0.85,
        "Bonds": 0.10,
        "Gold": 0.05
    }
}


# ============================================================
# 4. CALCULATE HISTORICAL PORTFOLIO RETURNS
# ============================================================

portfolio_returns = {}

for strategy, weights in strategies.items():

    portfolio_return = (

        returns["Stocks"]
        * weights["Stocks"]

        +

        returns["Bonds"]
        * weights["Bonds"]

        +

        returns["Gold"]
        * weights["Gold"]
    )

    portfolio_returns[strategy] = (
        portfolio_return
    )


portfolio_returns = pd.DataFrame(
    portfolio_returns
)


# ============================================================
# 5. CREATE MACHINE LEARNING FEATURES
# ============================================================

# The model uses previous market performance
# to predict the next portfolio return.

feature_data = pd.DataFrame(index=returns.index)

# Individual asset returns
for asset in assets:

    feature_data[
        asset + "_Return"
    ] = returns[asset]

# Rolling returns
for asset in assets:

    feature_data[
        asset + "_MA5"
    ] = returns[asset].rolling(5).mean()

    feature_data[
        asset + "_MA20"
    ] = returns[asset].rolling(20).mean()

# Volatility
for asset in assets:

    feature_data[
        asset + "_Volatility"
    ] = returns[asset].rolling(20).std()


# ============================================================
# 6. CREATE TARGET VALUE
# ============================================================

# Predict the 20-day cumulative portfolio return.

prediction_horizon = 20

target_data = pd.DataFrame(
    index=portfolio_returns.index
)

for strategy in strategies:

    target_data[strategy] = (
        portfolio_returns[strategy]
        .rolling(prediction_horizon)
        .sum()
        .shift(-prediction_horizon)
    )


# Combine features and targets
dataset = pd.concat(
    [
        feature_data,
        target_data
    ],
    axis=1
).dropna()


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

split_index = int(
    len(dataset) * 0.80
)

train_data = dataset.iloc[
    :split_index
]

test_data = dataset.iloc[
    split_index:
]


features = feature_data.columns


X_train = train_data[
    features
]

X_test = test_data[
    features
]


# ============================================================
# 8. TRAIN ONE MODEL FOR EACH STRATEGY
# ============================================================

models = {}

predictions = {}

actual_values = {}

print("\nTraining prediction models...\n")


for strategy in strategies:

    y_train = train_data[
        strategy
    ]

    y_test = test_data[
        strategy
    ]

    # Random Forest regression model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    models[strategy] = model

    # Predictions
    predicted = model.predict(
        X_test
    )

    predictions[strategy] = predicted

    actual_values[strategy] = (
        y_test.values
    )

    # Performance metrics
    mae = mean_absolute_error(
        y_test,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predicted
        )
    )

    print(
        f"{strategy:<15} "
        f"MAE: {mae:.5f} | "
        f"RMSE: {rmse:.5f}"
    )


# ============================================================
# 9. VALUE-EQUIVALENCE SCORE
# ============================================================

# Calculate predicted long-term return
# for every strategy.

predicted_mean_returns = {}

for strategy in strategies:

    predicted_mean_returns[strategy] = (
        np.mean(
            predictions[strategy]
        )
    )


print("\n" + "=" * 70)
print(" PREDICTED PORTFOLIO PERFORMANCE")
print("=" * 70)


for strategy in strategies:

    predicted_return = (
        predicted_mean_returns[strategy]
        * 100
    )

    print(
        f"{strategy:<15}: "
        f"{predicted_return:.2f}% "
        f"predicted 20-day return"
    )


# ============================================================
# 10. HISTORICAL PERFORMANCE
# ============================================================

historical_results = []


for strategy in strategies:

    strategy_returns = (
        portfolio_returns[strategy]
    )

    cumulative_return = (
        (1 + strategy_returns).prod()
        - 1
    )

    annual_return = (
        (1 + cumulative_return)
        ** (
            252 /
            len(strategy_returns)
        )
        - 1
    )

    volatility = (
        strategy_returns.std()
        * np.sqrt(252)
    )

    sharpe_ratio = (
        annual_return /
        volatility
        if volatility != 0
        else 0
    )

    historical_results.append({

        "Strategy": strategy,

        "Historical Return (%)":
            annual_return * 100,

        "Annual Volatility (%)":
            volatility * 100,

        "Sharpe Ratio":
            sharpe_ratio
    })


historical_results = pd.DataFrame(
    historical_results
)


# ============================================================
# 11. COMBINE PREDICTIONS AND HISTORICAL RESULTS
# ============================================================

historical_results[
    "Predicted 20-Day Return (%)"
] = [
    predicted_mean_returns[strategy]
    * 100
    for strategy in strategies
]


print("\n" + "=" * 70)
print(" PORTFOLIO COMPARISON")
print("=" * 70)

print(
    historical_results.to_string(
        index=False
    )
)


# ============================================================
# 12. VALUE-EQUIVALENCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print(" VALUE-EQUIVALENCE ANALYSIS")
print("=" * 70)


# Assume an initial investment
initial_investment = 100000


value_results = []


for strategy in strategies:

    predicted_return = (
        predicted_mean_returns[
            strategy
        ]
    )

    predicted_value = (
        initial_investment
        *
        (
            1 + predicted_return
        )
    )

    value_results.append({

        "Strategy": strategy,

        "Initial Investment":
            initial_investment,

        "Predicted Return (%)":
            predicted_return * 100,

        "Predicted Value":
            predicted_value
    })


value_results = pd.DataFrame(
    value_results
)


print(
    value_results.to_string(
        index=False
    )
)


# ============================================================
# 13. IDENTIFY BEST STRATEGY
# ============================================================

best_strategy = max(
    predicted_mean_returns,
    key=predicted_mean_returns.get
)


print("\n" + "=" * 70)
print(" BEST PREDICTED STRATEGY")
print("=" * 70)

print(
    "Strategy:",
    best_strategy
)

print(
    "Predicted Return:",
    f"{predicted_mean_returns[best_strategy] * 100:.2f}%"
)

print(
    "Predicted Value:",
    f"${value_results.loc[value_results['Strategy'] == best_strategy, 'Predicted Value'].iloc[0]:,.2f}"
)


# ============================================================
# 14. HISTORICAL CUMULATIVE PERFORMANCE GRAPH
# ============================================================

plt.figure(
    figsize=(12, 6)
)

for strategy in strategies:

    cumulative = (
        1
        +
        portfolio_returns[strategy]
    ).cumprod()

    plt.plot(
        cumulative,
        label=strategy
    )

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Growth of $1"
)

plt.title(
    "Historical Portfolio Performance"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 15. PREDICTED RETURN COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    list(predicted_mean_returns.keys()),
    [
        value * 100
        for value in
        predicted_mean_returns.values()
    ]
)

plt.xlabel(
    "Portfolio Strategy"
)

plt.ylabel(
    "Predicted 20-Day Return (%)"
)

plt.title(
    "Machine Learning Predicted Portfolio Performance"
)

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# 16. RISK-RETURN COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    historical_results[
        "Annual Volatility (%)"
    ],
    historical_results[
        "Historical Return (%)"
    ]
)

for i in range(
    len(historical_results)
):

    plt.annotate(
        historical_results.iloc[
            i
        ]["Strategy"],
        (
            historical_results.iloc[
                i
            ]["Annual Volatility (%)"],
            historical_results.iloc[
                i
            ]["Historical Return (%)"]
        )
    )

plt.xlabel(
    "Annual Volatility (%)"
)

plt.ylabel(
    "Historical Annual Return (%)"
)

plt.title(
    "Portfolio Risk-Return Comparison"
)

plt.grid(True)

plt.show()
