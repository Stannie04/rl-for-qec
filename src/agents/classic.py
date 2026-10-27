import ldpc
import numpy as np
import torch
import pymatching
import time

class ClassicAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.x_decoder, self.z_decoder = None, None

    def select_action(self, observation, evaluate=False):
        # Use belief propagation to decode the syndrome and determine the error

        x_syndrome = self.env.code.x_syndrome.cpu().numpy()
        z_syndrome = self.env.code.z_syndrome.cpu().numpy()

        start_time = time.perf_counter_ns()
        x_error = self.x_decoder.decode(z_syndrome)
        z_error = self.z_decoder.decode(x_syndrome)
        end_time = time.perf_counter_ns()

        # Z error indices are offset by the number of data qubits, so we need to add that offset
        x_error_index = np.argwhere(x_error).flatten()
        z_error_index = np.argwhere(z_error).flatten() + self.env.code.n_data

        return torch.from_numpy(np.concatenate((x_error_index, z_error_index))).long(), (end_time - start_time) / 1000  # Return time in microseconds


class MWPMAgent(ClassicAgent):
    def __init__(self, env, config):
        super().__init__(env, config)

        self.x_decoder = pymatching.Matching(env.code.H.z.cpu().numpy())
        self.z_decoder = pymatching.Matching(env.code.H.x.cpu().numpy())


class BPAgent(ClassicAgent):
    def __init__(self, env, config):
        super().__init__(env, config)

        H_x = np.array(env.code.H.x.cpu(), dtype=np.int8)
        H_z = np.array(env.code.H.z.cpu(), dtype=np.int8)

        if config.noise_model == "depolarizing":
            physical_error_rate = 2*env.curriculum_error_rate / 3
        elif config.noise_model == "bit_flip":
            physical_error_rate = env.curriculum_error_rate
        else:
            raise ValueError(f"Unknown noise model: {config.noise_model}")

        self.x_decoder = ldpc.BpDecoder(H_z, error_rate=physical_error_rate, max_iter=config.bp_iterations)
        self.z_decoder = ldpc.BpDecoder(H_x, error_rate=physical_error_rate, max_iter=config.bp_iterations)


class BPOSDAgent(ClassicAgent):
    def __init__(self, env, config):
        super().__init__(env, config)

        H_x = np.array(env.code.H.x.cpu(), dtype=np.int8)
        H_z = np.array(env.code.H.z.cpu(), dtype=np.int8)

        if config.noise_model == "depolarizing":
            physical_error_rate = 2*env.curriculum_error_rate / 3
        elif config.noise_model == "bit_flip":
            physical_error_rate = env.curriculum_error_rate
        else:
            raise ValueError(f"Unknown noise model: {config.noise_model}")

        self.x_decoder = ldpc.BpOsdDecoder(H_z, error_rate=physical_error_rate, max_iter=50)
        self.z_decoder = ldpc.BpOsdDecoder(H_x, error_rate=physical_error_rate, max_iter=50)


class BPLSDAgent(ClassicAgent):
    def __init__(self, env, config):
        super().__init__(env, config)

        H_x = np.array(env.code.H.x.cpu(), dtype=np.int8)
        H_z = np.array(env.code.H.z.cpu(), dtype=np.int8)

        if config.noise_model == "depolarizing":
            physical_error_rate = 2*env.curriculum_error_rate / 3
        elif config.noise_model == "bit_flip":
            physical_error_rate = env.curriculum_error_rate
        else:
            raise ValueError(f"Unknown noise model: {config.noise_model}")

        self.x_decoder = ldpc.BpLsdDecoder(H_z, error_rate=physical_error_rate, max_iter=30)
        self.z_decoder = ldpc.BpLsdDecoder(H_x, error_rate=physical_error_rate, max_iter=30)