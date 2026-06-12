import gymnasium as gym
import numpy as np
import torch
import galois

from src.read_config import ConfigParser
from src.environment.code import QLDPCCode

class QLDPCEnv(gym.Env):

    def __init__(self, config: ConfigParser, shots=None):
        super().__init__()
        self.device = config.device

        self.code = QLDPCCode(config)
        self.action_space = gym.spaces.Discrete(self.code.n_data)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.code.n_data,), dtype=np.int8)

        self.curriculum_error_rate = config.curriculum_start_error_rate
        self.curriculum_num_flips = 1

        self.episode_steps = 0
        self.max_episode_length = config.max_episode_length

        # Information about the previous step, used for reward calculation and logging.
        self.previous_num_errors = 0
        self.previous_num_syndromes = 0
        self.initial_errors = 0
        self.last_action = None
        self.correct_actions = []
        self.repeated_actions = []

        self.shots = shots
        self.num_shots = shots.shape[0] if shots is not None else 0
        self.shot_idx = 0

        self.info = None


    @property
    def observation(self):
        self.code.update_graph()
        return self.code.data.clone()


    def get_reward(self, actions):
        reward = 0.0

        reward += -1.0 if self.info["repeated_action"] else 0.0

        reward += 0.25 * (self.previous_num_syndromes - self.info["num_syndromes"])
        reward += 0.50 * (self.previous_num_errors - self.info["num_errors"]) / max(1, self.initial_errors)

        reward -= 0.01  # step penalty

        if self.info["error_free"]:
            reward += 2.0

        if self.info["logical_error"]:
            reward -= 2.0

        return reward


    @property
    def terminated(self):
        return self.info["error_free"] or self.info["logical_error"]


    @property
    def truncated(self):
        return self.episode_steps >= self.max_episode_length


    def update_info(self):
        self.info = {
            "episode_steps": self.episode_steps,
            "correct_actions": self.correct_actions,
            "repeated_actions": self.repeated_actions,
            "num_errors": self.code.num_x_errors,
            "num_syndromes": int(self.code.x_syndrome.sum() + self.code.z_syndrome.sum()),
            "error_free": self.code.is_error_free(),
            "logical_error": self.code.has_logical_error(),
            "repeated_action": np.array_equal(self.last_action, self.last_action),

            "logs": {
                    "Actions/Accuracy": np.mean(self.correct_actions[-100:]) if self.correct_actions else 0.0,
                    "Actions/Repeated Actions": np.mean(self.repeated_actions[-100:]) if self.repeated_actions else 0.0,
                    "Monitoring/Number of Errors": self.code.num_x_errors
                }
        }


    def step(self, action):
        actions = action.cpu().numpy()

        for action in actions:
            action = int(action)
            self.correct_actions.append(int(self._correct_action(action)))

            self.code.flip(action)

        self.update_info()

        reward = self.get_reward(actions)

        self.previous_num_errors = self.info["num_errors"]
        self.previous_num_syndromes = self.info["num_syndromes"]
        self.repeated_actions.append(self.info["repeated_action"])
        self.last_action = actions

        self.episode_steps += 1
        return self.observation, reward, self.terminated, self.truncated, self.info


    def reset(self, seed=None, options=None):

        if self.shots is not None:
            # error_pattern_x, error_pattern_z = self.shots[np.random.randint(self.num_shots)]
            error_pattern_x, error_pattern_z = self.shots[self.shot_idx]
            self.shot_idx += 1
            return self.reset_with_error_pattern(error_pattern_x, error_pattern_z)

        self.code.clear_errors()
        self.code.flip_set_number_of_qubits(1)
        self.code.flip_randomly(self.curriculum_error_rate)

        self._init_metrics_on_reset()

        return self.observation, self.info


    def reset_with_error_pattern(self, error_pattern_x, error_pattern_z):
        self.code.set_error_pattern(error_pattern_x, error_pattern_z)
        self._init_metrics_on_reset()

        return self.observation, self.info


    def render(self, mode='human'):
        self.code.render(mode=mode)


    def _correct_action(self, action):
        return self.code.x_errors[action] == 1  # Correct if the flipped qubit had an error


    def _init_metrics_on_reset(self):
        self.update_info()
        self.previous_num_syndromes = self.info["num_syndromes"]
        self.initial_errors = self.info["num_errors"]
        self.previous_num_errors = self.initial_errors
        self.episode_steps = 0
        self.last_action = None