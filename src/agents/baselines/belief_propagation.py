import ldpc
import numpy as np
import torch

class BPAgent:
    def __init__(self, env, config):
        self.env = env

        H_z = np.array(env.code.H_z.cpu(), dtype=np.int8)
        H_x = np.array(env.code.H_x.cpu(), dtype=np.int8)

        self.bp_x = ldpc.BpDecoder(H_x, error_rate=env.curriculum_error_rate, max_iter=100)
        self.bp_z = ldpc.BpDecoder(H_z, error_rate=env.curriculum_error_rate, max_iter=100)

    def select_action(self, observation, evaluate=False):
        # Use belief propagation to decode the syndrome and determine the error

        x_syndrome, z_syndrome = self.env.code.get_syndrome()
        x_error = self.bp_x.decode(x_syndrome.cpu().numpy())
        z_error = self.bp_z.decode(z_syndrome.cpu().numpy())

        x_error_index = np.argwhere(x_error == 1).flatten()
        z_error_index = np.argwhere(z_error == 1).flatten()

        return torch.tensor(z_error_index), torch.tensor(x_error_index)


class BPOSDAgent(BPAgent):
    def __init__(self, env, config):
        super().__init__(env, config)

        H_z = np.array(env.code.H_x.cpu(), dtype=np.int8)
        H_x = np.array(env.code.H_z.cpu(), dtype=np.int8)

        self.bp_x = ldpc.BpOsdDecoder(H_z, error_rate=env.curriculum_error_rate, max_iter=100)
        self.bp_z = ldpc.BpOsdDecoder(H_x, error_rate=env.curriculum_error_rate, max_iter=100)