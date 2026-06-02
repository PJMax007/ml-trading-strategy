"""
Market Simulator for ML Trading Strategy
Simulates portfolio performance based on trading signals.
"""

import pandas as pd
import numpy as np


def compute_portvals(prices, signals, start_val=100000, commission=9.95, impact=0.005):
    """
    Simulate portfolio value over time given trading signals.

    Parameters:
        prices     : pd.Series of adjusted closing prices
        signals    : pd.Series of BUY (1), SELL (-1), HOLD (0) signals
        start_val  : Starting portfolio cash value
        commission : Fixed commission per trade
        impact     : Market impact cost as fraction of share price

    Returns:
        pd.Series of daily portfolio values
    """
    cash = start_val
    holdings = 0
    port_vals = []

    for date in prices.index:
        price = prices[date]
        signal = signals.get(date, 0)

        if signal == 1 and holdings == 0:
            # BUY
            shares = int(cash / (price * (1 + impact)))
            cost = shares * price * (1 + impact) + commission
            cash -= cost
            holdings += shares

        elif signal == -1 and holdings > 0:
            # SELL
            proceeds = holdings * price * (1 - impact) - commission
            cash += proceeds
            holdings = 0

        port_val = cash + holdings * price
        port_vals.append(port_val)

    return pd.Series(port_vals, index=prices.index)


def get_portfolio_stats(port_vals, risk_free_rate=0.0):
    """
    Compute key portfolio performance metrics.

    Parameters:
        port_vals      : pd.Series of daily portfolio values
        risk_free_rate : Daily risk-free rate (default 0)

    Returns:
        dict with cumulative return, avg daily return,
        Sharpe ratio, and max drawdown
    """
    daily_returns = port_vals.pct_change().dropna()

    cumulative_return = (port_vals[-1] / port_vals[0]) - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    # Annualized Sharpe Ratio (252 trading days)
    sharpe_ratio = (
        np.sqrt(252) * (avg_daily_return - risk_free_rate) / std_daily_return
        if std_daily_return != 0 else 0.0
    )

    # Max Drawdown
    rolling_max = port_vals.cummax()
    drawdown = (port_vals - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "cumulative_return": round(cumulative_return, 4),
        "avg_daily_return": round(avg_daily_return, 6),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown": round(max_drawdown, 4),
    }


def print_stats(port_vals, label="Strategy"):
    """
    Print portfolio stats in a readable format.
    """
    stats = get_portfolio_stats(port_vals)
    print(f"\n--- {label} Performance ---")
    print(f"  Cumulative Return : {stats['cumulative_return']:.2%}")
    print(f"  Avg Daily Return  : {stats['avg_daily_return']:.4%}")
    print(f"  Sharpe Ratio      : {stats['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown      : {stats['max_drawdown']:.2%}")
