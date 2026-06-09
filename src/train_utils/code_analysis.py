from src.environment import QLDPCEnv
import math
import numpy as np


def probabilities_of_k_errors_per_shot(code):
    total_number_qubits = code.n_data

    error_probabilities = [0.001, 0.002, 0.003, 0.004, 0.005]

    for error_rate in error_probabilities:
        probabilities = []
        # for k in range(total_number_qubits + 1):
        for k in range(6):
            prob_k_errors = math.comb(total_number_qubits, k) * (error_rate ** k) * ((1 - error_rate) ** (total_number_qubits - k))
            probabilities.append(prob_k_errors)

        print(f"Error Rate: {error_rate:.3%}")
        for k, prob in enumerate(probabilities):
            print(f"  Probability of {k} errors: {prob:e}")

def analyze_datasets(config):

    sac_mistakes = np.load(f"datasets/mistakes_sac_{config.code_name}.npy", allow_pickle=True)
    bp_mistakes = np.load(f"datasets/mistakes_bp_{config.code_name}.npy", allow_pickle=True)
    bp_osd_mistakes = np.load(f"datasets/mistakes_bp_osd_{config.code_name}.npy", allow_pickle=True)

    for agent_name, mistakes in [("SAC", sac_mistakes), ("BP", bp_mistakes), ("BP+OSD", bp_osd_mistakes)]:

        values, counts = np.unique(mistakes[:, 0, :].sum(axis=-1), return_counts=True)
        print(f"\n\n{agent_name} Mistakes Distribution:")
        for v, c in zip(values, counts):
            print(f"  {v} errors: {c} samples")
        print(f"\nTotal samples: {len(mistakes)}")
        # Check if the other agents make the same mistakes
        for other_agent_name, other_mistakes in [("SAC", sac_mistakes), ("BP", bp_mistakes), ("BP+OSD", bp_osd_mistakes)]:
            if other_agent_name == agent_name:
                continue
            overlap = sum(any(np.array_equal(m, om) for om in other_mistakes) for m in mistakes)
            print(f"  Overlap with {other_agent_name}: {overlap} samples ({overlap/len(mistakes)*100:.2f}%)")


def full_code_analysis(config):
    env = QLDPCEnv(config)
    code = env.code
    print(f"Code Name: {config.code_name}")
    probabilities_of_k_errors_per_shot(code)
    analyze_datasets(config)