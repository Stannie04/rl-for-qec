import ldpc
import numpy as np
import torch

class BPAgent:
    def __init__(self, env, config):
        self.env = env

        H_z = np.array(env.code.H.z.cpu(), dtype=np.int8)
        H_x = np.array(env.code.H.x.cpu(), dtype=np.int8)

        self.bp_x = ldpc.BpDecoder(H_x, error_rate=env.curriculum_error_rate, max_iter=1_000)
        self.bp_z = ldpc.BpDecoder(H_z, error_rate=env.curriculum_error_rate, max_iter=1_000)

    def select_action(self, observation, evaluate=False):
        # Use belief propagation to decode the syndrome and determine the error

        # x_syndrome = observation[self.env.code.x_idx, 3].cpu().numpy()
        # z_syndrome = observation[self.env.code.z_idx, 3].cpu().numpy()

        x_syndrome = self.env.code.x_syndrome.cpu().numpy()
        z_syndrome = self.env.code.z_syndrome.cpu().numpy()

        x_error = self.bp_x.decode(x_syndrome)
        z_error = self.bp_z.decode(z_syndrome)

        x_error_index = np.argwhere(x_error == 1).flatten()
        z_error_index = np.argwhere(z_error == 1).flatten()

        return torch.tensor(z_error_index), torch.tensor(x_error_index)


class BPOSDAgent(BPAgent):
    def __init__(self, env, config):
        super().__init__(env, config)

        H_x = np.array(env.code.H.x.cpu(), dtype=np.int8)
        H_z = np.array(env.code.H.z.cpu(), dtype=np.int8)

        self.bp_x = ldpc.BpOsdDecoder(H_x, error_rate=env.curriculum_error_rate, max_iter=10_000)
        self.bp_z = ldpc.BpOsdDecoder(H_z, error_rate=env.curriculum_error_rate, max_iter=10_000)


class BPLSDAgent(BPAgent):
    def __init__(self, env, config):
        super().__init__(env, config)

        H_x = np.array(env.code.H.x.cpu(), dtype=np.int8)
        H_z = np.array(env.code.H.z.cpu(), dtype=np.int8)

        self.bp_x = ldpc.BpLsdDecoder(H_x, error_rate=env.curriculum_error_rate, max_iter=30)
        self.bp_z = ldpc.BpLsdDecoder(H_z, error_rate=env.curriculum_error_rate, max_iter=30)