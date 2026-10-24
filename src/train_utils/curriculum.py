import numpy as np

class CurriculumScheduler:
    def __init__(self, config):
        self.noise_model = config.noise_model
        self.start_error_rate = config.curriculum_start_error_rate_144 * 144 / config.n
        self.end_error_rate = config.curriculum_end_error_rate_144 * 144 / config.n
        self.warmup_steps = config.curriculum_warmup_steps

    def progress(self, step):
        return np.clip(step / max(1, self.warmup_steps), 0, 1)

    def step(self, env, step):
        p = self.progress(step)
        env.curriculum_error_rate = self.start_error_rate + p * (self.end_error_rate - self.start_error_rate)

    def error_rates_for_steps(self, steps):
        return self.start_error_rate + self.progress(np.array(steps)) * (self.end_error_rate - self.start_error_rate)