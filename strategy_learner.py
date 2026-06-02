"""
Random Forest Strategy Learner for ML Trading Strategy
Predicts BUY/SELL/HOLD signals using supervised learning on technical indicators.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from indicators import get_all_indicators


# Action constants
BUY = 1
SELL = -1
HOLD = 0


class StrategyLearner:
    """
    A supervised ML trading strategy using a Random Forest classifier.

    Parameters:
        n_estimators : Number of trees in the Random Forest
        window       : Lookahead window to generate training labels
        threshold    : Return threshold to classify BUY/SELL vs HOLD
    """

    def __init__(self, n_estimators=100, window=5, threshold=0.02):
        self.n_estimators = n_estimators
        self.window = window
        self.threshold = threshold
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42
        )

    def _generate_labels(self, prices):
        """
        Generate trading labels based on future N-day return.
        BUY  (+1) if future return >  threshold
        SELL (-1) if future return < -threshold
        HOLD ( 0) otherwise
        """
        future_return = prices.shift(-self.window) / prices - 1
        labels = pd.Series(HOLD, index=prices.index)
        labels[future_return > self.threshold] = BUY
        labels[future_return < -self.threshold] = SELL
        return labels

    def add_evidence(self, prices):
        """
        Train the Random Forest model on historical price data.

        Parameters:
            prices : pd.Series of adjusted closing prices
        """
        indicators = get_all_indicators(prices)
        labels = self._generate_labels(prices)

        # Align and drop NaN rows
        data = indicators.copy()
        data["label"] = labels
        data.dropna(inplace=True)

        X = data[["momentum", "bbp", "rsi"]].values
        y = data["label"].values

        self.model.fit(X, y)

    def test_policy(self, prices):
        """
        Generate trading signals for a given price series.

        Returns:
            pd.Series of signals: BUY (1), SELL (-1), HOLD (0)
        """
        indicators = get_all_indicators(prices)
        indicators.dropna(inplace=True)

        X = indicators[["momentum", "bbp", "rsi"]].values
        predictions = self.model.predict(X)

        signals = pd.Series(predictions, index=indicators.index)
        return signals
