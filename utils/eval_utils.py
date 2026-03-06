from stable_baselines3 import SAC
from environments import MultivariateBicycleCode

def render_evaluation_episode(code_config, model_checkpoint, max_episode_steps=100):

    model = SAC.load(model_checkpoint)

    env = MultivariateBicycleCode(**code_config, evaluation_mode=True)

    obs, info = env.reset()
    env.render()

    for step in range(max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: Reward={reward}, Terminated={terminated}, Truncated={truncated}, Info={info}")
        env.render()

        if terminated or truncated:
            print(f"Episode finished after {step+1} steps with reward {reward} and info {info}")
            return

    print("Episode finished without termination or truncation.")