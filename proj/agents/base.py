from abc import abstractmethod
import numpy as np

class Agent:
    @abstractmethod
    def act(self, obs) -> np.ndarray:
        pass

class ZeroAgent(Agent):
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return np.zeros(self.env.action_dim)

class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return np.random.randn(self.env.action_dim)


class CorrelatedNoiseAgent(Agent):
    """Agent with temporally correlated actions (Ornstein-Uhlenbeck process).
    Much better for learning dynamics than pure random noise."""
    
    def __init__(self, env, theta=0.15, sigma=0.3):
        self.env = env
        self.theta = theta  # Mean reversion rate
        self.sigma = sigma  # Volatility
        self.action_prev = np.zeros(env.action_dim)
    
    def act(self, obs):
        # Ornstein-Uhlenbeck process: smoothly varying actions
        noise = np.random.randn(self.env.action_dim)
        self.action_prev = (
            self.action_prev 
            - self.theta * self.action_prev 
            + self.sigma * noise
        )
        return self.action_prev.copy()


class StructuredExplorationAgent(Agent):
    """Agent that cycles through different exploration patterns.
    Provides rich, diverse temporal structure for learning."""
    
    def __init__(self, env, pattern_length=50):
        self.env = env
        self.pattern_length = pattern_length
        self.step_count = 0
        self.current_mode = 0
        self.base_freq = 0.0
    
    def act(self, obs):
        t = self.step_count / self.pattern_length
        
        # Switch between different patterns every pattern_length steps
        if self.step_count % self.pattern_length == 0:
            self.current_mode = np.random.randint(4)
            self.base_freq = np.random.uniform(0.5, 3.0)
        
        action = np.zeros(self.env.action_dim)
        
        if self.current_mode == 0:
            # Sinusoidal with random frequency
            action[0] = 2.0 * np.sin(2 * np.pi * self.base_freq * t)
        elif self.current_mode == 1:
            # Square wave (sudden changes)
            action[0] = 2.0 if (int(4 * t) % 2 == 0) else -2.0
        elif self.current_mode == 2:
            # Linear ramp
            action[0] = 4.0 * (t % 0.5) - 1.0
        else:
            # Smooth random walk
            action[0] = np.clip(
                getattr(self, 'prev_action', 0.0) + np.random.randn() * 0.3,
                -2.0, 2.0
            )
        
        self.prev_action = action[0]
        self.step_count += 1
        return action


class MixedExplorationAgent(Agent):
    """Combines multiple exploration strategies with random switching.
    Best balance between exploration and learnability."""
    
    def __init__(self, env, switch_prob=0.05):
        self.env = env
        self.switch_prob = switch_prob
        self.correlated = CorrelatedNoiseAgent(env, theta=0.1, sigma=0.4)
        self.structured = StructuredExplorationAgent(env, pattern_length=40)
        self.use_structured = True
    
    def act(self, obs):
        # Randomly switch between structured and correlated exploration
        if np.random.random() < self.switch_prob:
            self.use_structured = not self.use_structured
        
        if self.use_structured:
            return self.structured.act(obs)
        else:
            return self.correlated.act(obs)
