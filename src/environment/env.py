import gymnasium as gym
import numpy as np
import galois
from src.read_config import ConfigParser
from src.environment.code import QECCode

class QECEnv(gym.Env):

    def __init__(self, config: ConfigParser, shots=None, evaluate=False):
        super().__init__()
        self.device = config.device
        self.code = QECCode(config)
        num_actions = self.code.n_data*2 if config.noise_model == "depolarizing" else self.code.n_data
        self.action_space = gym.spaces.Discrete(num_actions)
        self.evaluate = evaluate
        # self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.code.n_data,), dtype=np.int8)

        self.curriculum_error_rate = config.curriculum_start_error_rate_144 * 144 / self.code.n_data

        self.episode_steps = 0
        self.max_episode_length = config.max_episode_length if not evaluate else 2*self.code.d # Allow more steps during evaluation to give the agent a better chance to correct errors.

        # Information about the previous step, used for reward calculation and logging.
        self.previous_num_x_errors = 0
        self.previous_num_z_errors = 0
        self.previous_num_x_syndromes = 0
        self.previous_num_z_syndromes = 0
        self.initial_x_errors = 0
        self.initial_z_errors = 0
        self.initial_x_syndromes = 0
        self.initial_z_syndromes = 0

        self.actions_taken = []
        self.correct_actions = []
        self.repeated_actions = []

        self.shots = shots
        self.num_shots = shots.shape[0] if shots is not None else 0
        self.shot_idx = 0

        self.info = None


    @property
    def observation(self):
        self.code.update_graph(self.curriculum_error_rate)
        return self.code.data.clone()


    def get_reward(self, actions):
        reward = -0.05

        reward += -0.3 if self.info["repeated_action"] else 0.0

        x_syndrome_delta = self.previous_num_x_syndromes - self.info["num_x_syndromes"]
        z_syndrome_delta = self.previous_num_z_syndromes - self.info["num_z_syndromes"]
        reward += 0.5 * x_syndrome_delta / self.code.num_x_checks
        reward += 0.5 * z_syndrome_delta / self.code.num_z_checks

        if self.info["syndrome_free"] and not self.info["logical_error"]:
            reward += 5

        elif self.truncated:
            reward -= 5.0

        return reward


    @property
    def terminated(self):
        return self.info["syndrome_free"] or self.info["logical_error"]


    @property
    def truncated(self):
        return self.episode_steps >= self.max_episode_length


    def update_info(self):
        self.info = {
            "episode_steps": self.episode_steps,
            "repeated_actions": self.repeated_actions,
            "num_x_errors": self.code.num_x_errors,
            "num_z_errors": self.code.num_z_errors,
            "num_x_syndromes": int(self.code.x_syndrome.sum()),
            "num_z_syndromes": int(self.code.z_syndrome.sum()),
            # "error_free": self.code.is_error_free(),
            "syndrome_free" : self.code.is_syndrome_free(),
            "logical_error": self.code.has_logical_error(),
            "repeated_action": self.actions_taken[-1] == self.actions_taken[-2] if len(self.actions_taken) > 1 else False,

            "logs": {
                    "Actions/Repeated Actions": np.mean(self.repeated_actions[-100:]) if self.repeated_actions else 0.0,
                    "Monitoring/Number of Errors": self.code.num_x_errors + self.code.num_z_errors
                }
        }


    def step(self, action):

        action = int(action.cpu().numpy())
        self.code.flip(action)

        self.episode_steps += 1
        self.actions_taken.append(action)

        self.update_info()
        reward = self.get_reward(action)

        self.previous_num_x_errors = self.info["num_x_errors"]
        self.previous_num_z_errors = self.info["num_z_errors"]
        self.previous_num_x_syndromes = self.info["num_x_syndromes"]
        self.previous_num_z_syndromes = self.info["num_z_syndromes"]
        self.repeated_actions.append(self.info["repeated_action"])

        return self.observation, reward, self.terminated, self.truncated, self.info


    def reset(self, seed=None, options=None):

        if self.shots is not None:
            error_pattern_x, error_pattern_z = self.shots[self.shot_idx]
            self.shot_idx += 1
            return self.reset_with_error_pattern(error_pattern_x, error_pattern_z)

        self.code.clear_errors()
        self.code.flip_randomly(self.curriculum_error_rate)

        self._init_metrics_on_reset()

        return self.observation, self.info


    def reset_with_error_pattern(self, error_pattern_x, error_pattern_z):
        self.code.set_error_pattern(error_pattern_x, error_pattern_z)
        self._init_metrics_on_reset()

        return self.observation, self.info


    def render(self, mode='human', title=None, return_fig=False):
        return self.code.render(mode, title=title, return_fig=return_fig)


    def _init_metrics_on_reset(self):
        self.episode_steps = 0
        self.actions_taken = []
        self.correct_actions = []
        self.update_info()
        self.previous_num_x_syndromes = self.info["num_x_syndromes"]
        self.previous_num_z_syndromes = self.info["num_z_syndromes"]
        self.initial_x_errors = self.info["num_x_errors"]
        self.initial_z_errors = self.info["num_z_errors"]
        self.initial_x_syndromes = self.info["num_x_syndromes"]
        self.initial_z_syndromes = self.info["num_z_syndromes"]
        self.previous_num_x_errors = self.initial_x_errors
        self.previous_num_z_errors = self.initial_z_errors