from environments import MultivariateBicycleCode
from agents import RandomAgent

if __name__ == '__main__':
    # As per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
    env = MultivariateBicycleCode(l=12, m=6, interaction_vectors=[(3, 6), (6, -3)])
    agent = RandomAgent(env)

    observation, info = env.reset()
    for i in range(10):
        observation, reward, terminated, truncated, info = env.step(agent.select_action(observation))
        env.render()