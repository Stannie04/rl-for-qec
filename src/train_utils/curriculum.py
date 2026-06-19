import numpy as np

class CurriculumScheduler:
    def __init__(self, config):
        self.start_error_rate = config.curriculum_start_error_rate
        self.end_error_rate = config.curriculum_end_error_rate
        self.start_weight = config.curriculum_start_weight
        self.end_weight = config.curriculum_end_weight
        self.warmup_steps = config.curriculum_warmup_steps

    def progress(self, step):
        return min(1.0, step / max(1, self.warmup_steps))

    def step(self, env, step):
        p = self.progress(step)
        env.curriculum_error_rate = self.start_error_rate + p * (self.end_error_rate - self.start_error_rate)
        env.curriculum_num_flips = int(round(self.start_weight + p * (self.end_weight - self.start_weight)))

    def error_rates_for_steps(self, steps):
        return np.array([self.start_error_rate + self.progress(step) * (self.end_error_rate - self.start_error_rate) for step in steps])