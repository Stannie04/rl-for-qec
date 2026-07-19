import pymatching
import numpy as np
import torch

class MWPMAgent:
    def __init__(self, env, config):
        self.env = env

        self.matching_x = pymatching.Matching(env.code.H_x.cpu().numpy())
        self.matching_z = pymatching.Matching(env.code.H_z.cpu().numpy())

    def select_action(self, observation, evaluate=False):

        x_syndrome = self.env.code.x_syndrome.cpu().numpy().astype(np.uint8)
        z_syndrome = self.env.code.z_syndrome.cpu().numpy().astype(np.uint8)

        x_error = self.matching_x.decode(x_syndrome)
        z_error = self.matching_z.decode(z_syndrome)

        x_error_index = np.argwhere(x_error).flatten()
        z_error_index = np.argwhere(z_error).flatten()

        return torch.tensor(z_error_index), torch.tensor(x_error_index)