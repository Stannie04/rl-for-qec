from environments import MultivariateBicycleCode
from agents import RandomAgent

if __name__ == '__main__':
    # As per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
    env = MultivariateBicycleCode(l=12, m=6, interaction_vectors=[(3, 6), (6, -3)])

    env.init_errors(num_errors=1)
    env.update_stabilizers()
    print(env)
    print(f"Number of errors: {env.num_errors()}")
    print(f"Number of syndromes: {env.num_syndromes()}")