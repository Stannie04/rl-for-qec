from src.environment import QLDPCEnv
import math

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


def full_code_analysis(config):
    env = QLDPCEnv(config)
    code = env.code
    print(f"Code Name: {config.code_name}")
    probabilities_of_k_errors_per_shot(code)