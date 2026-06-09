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

    @property
    def observation(self):
        return self.code.data.clone()

    def get_reward(self, action, num_syndromes, num_errors):
        reward = 0.0

        reward += -1.0 if action == self.last_action else 0.0

        reward += 0.25 * (self.previous_num_syndromes - num_syndromes)
        reward += 0.50 * (self.previous_num_errors - num_errors) / max(1, self.initial_errors)

        reward -= 0.01  # step penalty

        if self.code.is_error_free():
            reward += 2.0

        if self.code.has_logical_error():
            reward -= 2.0

        return reward


    @property
    def terminated(self):
        return self.code.is_error_free() or self.code.has_logical_error()


    @property
    def truncated(self):
        return self.episode_steps >= self.max_episode_length


    def _get_info(self):
        num_errors = self.code.x_errors.sum().item()
        return{
            "episode_steps": self.episode_steps,
            "correct_actions": self.correct_actions,
            "repeated_actions": self.repeated_actions,
            "num_errors": num_errors,
            # "errors": torch.where(self.code.x_errors == 1)[0].cpu().numpy(),
            "logs": {
                    "Actions/Accuracy": np.mean(self.correct_actions[-100:]) if self.correct_actions else 0.0,
                    "Actions/Repeated Actions": np.mean(self.repeated_actions[-100:]) if self.repeated_actions else 0.0,
                    "Monitoring/Number of Errors": num_errors
                }
        }


    def step(self, action):
        reward = torch.tensor(0.0).to(self.device)
        actions = action.cpu().numpy()

        for action in actions:
            action = int(action)

            self.correct_actions.append(int(self._correct_action(action)))

            self.code.flip(action)

            x_syndrome, z_syndrome = self.code.get_syndrome()
            num_syndromes = x_syndrome.sum()
            num_errors = self.code.x_errors.sum()

            reward = self.get_reward(action, num_syndromes, num_errors)

            self.previous_num_errors = num_errors
            self.previous_num_syndromes = num_syndromes

            self.repeated_actions.append(1 if self.last_action == action else 0)
            self.last_action = action

        self.episode_steps += 1
        self.code.update_graph()
        return self.observation, reward, self.terminated, self.truncated, self._get_info()


    def reset(self, seed=None, options=None):
        self.code.x_errors.zero_()
        self.code.z_errors.zero_()

        self.episode_steps = 0
        self.last_action = None

        if self.shots is not None:
            # Pop shot from the list
            shot = self.shots.pop(0)
            self.code.set_error_pattern(shot[0], shot[1])
        else:
            # Ensure at least one error is present at the start of each episode to combat no-op bias
            self.code.flip_set_number_of_qubits(1)
            self.code.flip_randomly(self.curriculum_error_rate)

        self.code.update_graph()

        x_syndrome, z_syndrome = self.code.get_syndrome()
        self.previous_num_syndromes = int(x_syndrome.sum().item() + z_syndrome.sum().item())
        self.previous_num_errors = self.code.x_errors.sum().item()
        self.initial_errors = self.code.x_errors.sum().item()

        return self.observation, self._get_info()


    def render(self, mode='human'):
        self.code.render(mode=mode)


    def _correct_action(self, action):
        if action == self.code.no_op_index:
            return self.code.is_error_free()  # No-op is correct if there are no errors
        else:
            return self.code.x_errors[action] == 1  # Correct if the flipped qubit had an error

