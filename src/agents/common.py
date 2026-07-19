import torch

def get_action_mask(state, env):
    # Based on encoding:
    # [is_qubit, is_x_syndrome, is_z_syndrome, x_syndrome, z_syndrome]
    # mask = H_x_T @ x_syndrome + H_z_T @ z_syndrome
    # Gets only qubits that are connected to active syndromes

    x_mask = env.code.H_x_T @ state.x[env.code.x_idx, 3]
    z_mask = env.code.H_z_T @ state.x[env.code.z_idx, 4]

    return (x_mask + z_mask) > 0


def get_qubit_mask(state):
    return torch.where(state.x[:,0] > 0.5)[0]