"""
Technical Indicators for ML Trading Strategy
Computes momentum, Bollinger Bands, and RSI from price data.
"""

import pandas as pd
import numpy as np


def get_momentum(prices, window=10):
    """
    Momentum: ratio of current price to price N days ago.
    momentum[t] = (price[t] / price[t - window]) - 1
    """
    momentum = (prices / prices.shift(window)) - 1
    return momentum


def get_bollinger_bands(prices, window=20):
    """
    Bollinger Band percentage:
    bbp[t] = (price[t] - lower_band[t]) / (upper_band[t] - lower_band[t])
    Value of 0 = lower band, 1 = upper band
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    bbp = (prices - lower_band) / (upper_band - lower_band)
    return bbp


def get_rsi(prices, window=14):
    """
    Relative Strength Index (RSI):
    Measures speed and magnitude of price changes.
    RSI > 70 = overbought, RSI < 30 = oversold
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_all_indicators(prices, window_mom=10, window_bb=20, window_rsi=14):
    """
    Returns a DataFrame with all three indicators as columns.
    """
    indicators = pd.DataFrame(index=prices.index)
    indicators["momentum"] = get_momentum(prices, window_mom)
    indicators["bbp"] = get_bollinger_bands(prices, window_bb)
    indicators["rsi"] = get_rsi(prices, window_rsi)
    return indicators
