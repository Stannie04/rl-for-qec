import time
from tqdm import tqdm
import torch
from src.environment import QLDPCEnv
from src.agents import DQNAgent, SACAgent, MoEAgent, BPAgent, BPOSDAgent
from src.train_utils import load_shots

def get_physical_error_rate(config, step):
    return config.moe_start_p + (step / config.moe_num_timesteps) * (config.moe_end_p - config.moe_start_p)


def moe_training_loop(config, env, agent, expert_list):
    start_time = time.time()

    successes = 0
    for step in range(config.moe_num_timesteps):
        error_rate = get_physical_error_rate(config, step)
        env.curriculum_error_rate = error_rate
        obs, info = env.reset()
        action, log_prob = agent.select_action(obs)
        decoder = expert_list[action]
        reward = decoder_inference(config, decoder, env, obs)
        loss = agent.update(log_prob, reward)

        print(f"Step {step+1}/{config.moe_num_timesteps}, p: {error_rate:.4f}, Action: {action}, Reward: {reward:.4f}, Loss: {loss:.4f}")


    print(f"MoE Training completed in {time.time() - start_time:.2f} seconds. Success rate: {successes / config.moe_num_timesteps:.4f}")


def decoder_inference(config, agent, env, obs):

    start_time = time.time()
    done = False
    while not done:
        action, _ = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    end_time = time.time()
    return env.code.is_error_free() - config.moe_time_penalty_factor * (end_time - start_time)


def get_decoders(config, env):
    expert_list = []

    for decoder in config.moe_experts:
        match decoder:
            case "bp": expert_list.append(BPAgent(env, config))
            case "bp_osd": expert_list.append(BPOSDAgent(env, config))
            case "sac":
                agent = SACAgent(env, config)
                checkpoint = torch.load(f"checkpoints/{config.agent_name}_{config.code_name}.pt", map_location=config.device)
                agent.actor.load_state_dict(checkpoint["actor"])
                agent.critic1.load_state_dict(checkpoint["critic1"])
                agent.critic2.load_state_dict(checkpoint["critic2"])
                expert_list.append(agent)
            case _: raise ValueError(f"Unknown decoder: {decoder}")

    return expert_list


def train_moe(config):
    shots = load_shots(config, dataset_type="random_nonzero")
    env = QLDPCEnv(config, shots)
    agent = MoEAgent(config, env)
    expert_list = get_decoders(config, env)

    moe_training_loop(config, env, agent, expert_list)