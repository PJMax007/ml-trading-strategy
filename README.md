# ML Trading Strategy — Random Forest & Q-Learning

A machine learning project implementing two algorithmic trading strategies using supervised and reinforcement learning techniques, developed as part of the Georgia Tech OMSCS AI curriculum.

---

## Overview

This project explores the use of machine learning to make portfolio trading decisions based on historical equity data. Two distinct strategies were designed, trained, and backtested:

- **Strategy Learner** — Supervised ML approach using a Random Forest classifier
- **Q-Learner** — Reinforcement learning agent using Q-Learning to optimize buy/sell/hold decisions

---

## Strategies

### 1. Random Forest Strategy Learner
- Uses technical indicators as features (e.g., momentum, Bollinger Bands, RSI)
- Trains a Random Forest classifier on historical price/volume data
- Predicts trading signals: BUY, SELL, or HOLD
- Evaluated against a benchmark buy-and-hold strategy

### 2. Q-Learning Agent
- Models the trading problem as a Markov Decision Process (MDP)
- State space: discretized technical indicators
- Actions: BUY, SELL, HOLD
- Reward function: daily portfolio return
- Learns optimal policy through
