from abc import abstractmethod
from math import sin

import numpy as np
import torch

class Env:
    name: str

    def __init__(self):
        pass

    @abstractmethod
    def _reset(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def _step(self, state, action) -> tuple[np.ndarray, np.ndarray]:
        """Internal step function to be implemented by subclasses."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        pass

    def reset(self):
        """Returns: initial_state, initial_observed_state"""
        s, o = self._reset()
        return self._o(s, o)

    def step(self, state, action):
        """Returns: next_state, next_observed_state"""
        s, a = self._a(state, action)
        return self._o(*self._step(s, a))

    def _a(self, state, action):
        assert state["env_name"] == self.name
        return state["value"], action

    def _o(self, next_state: np.ndarray, next_obs: np.ndarray):
        return {
            "env_name": self.name,
            "type": "next_state",
            "value": next_state
        }, {
            "env_name": self.name,
            "type": "next_observed_state",
            "value": next_obs
        }


class DummyEnv(Env):
    """A dummy environment for testing purposes."""

    factors: np.ndarray
    STATE_DIM = 10
    HISTORY_LENGTH = 20
    history = np.zeros((HISTORY_LENGTH, STATE_DIM))

    def __init__(self):
        super().__init__()
        self.name = "DummyEnv"
        self.factors = torch.randn(self.STATE_DIM, self.STATE_DIM).numpy()

    def _reset(self):
        self.history = np.zeros((self.HISTORY_LENGTH, self.STATE_DIM))
        return np.zeros(self.STATE_DIM), np.zeros(self.STATE_DIM)
        

    def _step(self, state, action):
        # State should simply be a mathematical operation on the action + some minor gaussian noise
        # Factors is a list of I vectors of length J = state_dim
        # y_i = state dot factors_I + action + 10 * state * sin(action) + small noise
        noise_factor = 0.03

        y = np.zeros(self.STATE_DIM)
        for i in range(self.STATE_DIM):
            y[i] = (
                np.dot(state, self.factors[i])
                + action[0]
                + 10 * state[i] * sin(action[0])
                + noise_factor * np.random.randn()
            ) * 0.01

        for j in range(self.HISTORY_LENGTH):
            #y += np.tanh(self.history[j]) * (.05 ** j)
            pass

        for j in range(self.HISTORY_LENGTH - 1, 0, -1):
            self.history[j] = self.history[j - 1]
        self.history[0] = y.copy()

        return y, y
    
    @property
    def action_dim(self) -> int:
        return 1
    
    @property
    def state_dim(self) -> int:
        return self.STATE_DIM
    
    @property
    def obs_dim(self) -> int:
        return self.STATE_DIM
