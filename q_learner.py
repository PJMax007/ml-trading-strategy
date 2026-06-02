"""
Q-Learning Agent for ML Trading Strategy
Learns optimal BUY/SELL/HOLD policy through reinforcement learning.
"""

import numpy as np


class QLearner:
    """
    A tabular Q-Learning agent.

    Parameters:
        num_states   : Number of discrete states in the environment
        num_actions  : Number of possible actions (3: BUY, SELL, HOLD)
        alpha        : Learning rate
        gamma        : Discount factor (future reward weight)
        rar          : Random action rate (exploration)
        radr         : Random action decay rate
        dyna         : Number of Dyna-Q hallucinated steps (0 = disabled)
    """

    def __init__(
        self,
        num_states=100,
        num_actions=3,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        # Q-table initialized to small random values
        self.q_table = np.random.uniform(low=0.0, high=0.01,
                                          size=(num_states, num_actions))

        self.state = 0
        self.action = 0

        # Dyna-Q experience replay memory
        self.dyna_memory = []

    def query_set_state(self, state):
        """
        Set the initial state. Returns a random action.
        """
        self.state = state
        if np.random.random() < self.rar:
            self.action = np.random.randint(self.num_actions)
        else:
            self.action = np.argmax(self.q_table[state])
        return self.action

    def query(self, next_state, reward):
        """
        Update Q-table using the Bellman equation and return next action.

        Q[s, a] = (1 - alpha) * Q[s, a]
                + alpha * (reward + gamma * max(Q[s', :]))
        """
        # Bellman update
        best_future = np.max(self.q_table[next_state])
        self.q_table[self.state, self.action] = (
            (1 - self.alpha) * self.q_table[self.state, self.action]
            + self.alpha * (reward + self.gamma * best_future)
        )

        # Dyna-Q: hallucinate additional experiences
        if self.dyna > 0:
            self.dyna_memory.append((self.state, self.action,
                                     next_state, reward))
            if len(self.dyna_memory) > 0:
                indices = np.random.randint(0, len(self.dyna_memory),
                                            size=self.dyna)
                for idx in indices:
                    s, a, ns, r = self.dyna_memory[idx]
                    best_f = np.max(self.q_table[ns])
                    self.q_table[s, a] = (
                        (1 - self.alpha) * self.q_table[s, a]
                        + self.alpha * (r + self.gamma * best_f)
                    )

        # Decay exploration rate
        self.rar *= self.radr

        # Choose next action
        if np.random.random() < self.rar:
            self.action = np.random.randint(self.num_actions)
        else:
            self.action = np.argmax(self.q_table[next_state])

        self.state = next_state
        return self.action
