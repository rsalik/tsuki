import gymnasium as gym
import numpy as np
from .base import Env

class GymEnv(Env):
    """A wrapper for Gymnasium environments."""

    g_env: gym.Env

    def __init__(self, env_name: str):
        super().__init__()
        self.name = f"GymEnv: {env_name}"
        self.g_env = gym.make(env_name)

    def _reset(self):
        obs, info = self.g_env.reset()
        # Ensure obs is a numpy array and flatten if needed
        obs = np.asarray(obs).flatten()
        return obs, obs 

    def _step(self, state, action):
        # Ensure action is in the correct format for the gym environment
        action = np.asarray(action).flatten()
        
        next_state, reward, terminated, truncated, info = self.g_env.step(action)
        # Ensure next_state is a numpy array and flatten if needed
        next_state = np.asarray(next_state).flatten()
        return next_state, next_state
    
    @property
    def action_dim(self) -> int:
        space = self.g_env.action_space
        if hasattr(space, 'shape') and space.shape:
            return int(np.prod(space.shape))
        return 1
    
    @property
    def state_dim(self) -> int:
        space = self.g_env.observation_space
        if hasattr(space, 'shape') and space.shape:
            return int(np.prod(space.shape))
        return 1
    
    @property
    def obs_dim(self) -> int:
        return self.state_dim