import time
from tqdm import tqdm
import os
import torch
from src.environment import QLDPCEnv
from src.agents import RouterAgent
from src.train_utils import load_shots, get_agent_and_inference, parallel_inference


def rl_train_loop(config, env, agent, expert_list, inference_list, train=True):
    start_time = time.time()

    checkpoint_dir = f"checkpoints/moe_{config.code_name}_{config.wandb_run_name}_{int(time.time())}"
    if train:
        os.mkdir(checkpoint_dir)

    moe_correct = 0
    losses = 0
    bp_correct = 0
    action_dist = 0

    results = {}

    for step in tqdm(range(config.moe_num_timesteps)):
    # for step, _ in tqdm(enumerate(env.shots)):
            # error_rate = get_physical_error_rate(config, step)
            # env.curriculum_error_rate = error_rate
            obs, info = env.reset()
            pattern = env.code.number_of_overlapping_stabilizers()
            results.setdefault(pattern, {"hits": 0, "successes": 0, "action_dist": 0})

            action, log_prob = agent.select_action(obs, evaluate=not train)

            if train:
                decoder = expert_list[0] # For now, just use BP always
                inference_fn = inference_list[0] # For now, just use BP always
            else:
                decoder = expert_list[action]
                inference_fn = inference_list[action]

            success = inference_fn(decoder, env, (obs, info))

            if train:
                reward = 1 if action == success else 0
                loss = agent.update(log_prob, reward)
                losses += loss
            else:
                reward = 1 if success else 0

            moe_correct += 1 if reward > 0 else 0
            bp_correct += 1 if success else 0
            action_dist += 1 if action == 0 else 0

            results[pattern]["hits"] += 1
            results[pattern]["successes"] += 1 if reward > 0 else 0
            # results[pattern]["bp_successes"] += 1 if (reward > 0 and action == 0) or (reward < 0 and action != 0) else 0
            results[pattern]["action_dist"] += 1 if action == 0 else 0

            if train and step % 100 == 0:
                print(f"Step {step+1}/{config.moe_num_timesteps}, MOE success rate: {moe_correct / 100}, BP success rate: {bp_correct / 100}%, BP chosen: {action_dist / 100}%")
                save_path = os.path.join(checkpoint_dir, f"moe_{step+1}_{moe_correct}.pt")
                agent.save(save_path)
                losses = 0
                moe_correct = 0
                action_dist = 0
                bp_correct = 0

            # print(f"Step {step+1}/{config.moe_num_timesteps}, Action: {action}, Reward: {reward:.4f}, Loss: {loss:.4f}")

    # print(f"MOE success rate: {successes / config.moe_num_timesteps}, BP success rate: {hits / config.moe_num_timesteps}%, BP chosen: {action_dist / config.moe_num_timesteps}%")
    print(f"MoE Training completed in {time.time() - start_time:.2f} seconds. Success rate: {moe_correct / config.moe_num_timesteps:.4f}")

    for pattern, pattern_results in results.items():
        total = pattern_results["hits"]
        moe_success_rate = pattern_results["successes"] / total if total > 0 else 0
        action_dist_rate = pattern_results["action_dist"] / total if total > 0 else 0
        print(f"Pattern: {pattern}, Hits: {total}, MOE Success Rate: {moe_success_rate:.4f}, Action distribution (BP chosen): {action_dist_rate:.4f}")


def get_decoders(config, env):
    expert_list, inference_list = [], []
    for decoder_name in config.moe_experts:
        decoder, inference = get_agent_and_inference(config, env, decoder_name)
        expert_list.append(decoder)
        inference_list.append(inference)

    return expert_list, inference_list


def evaluate_moe(config):
    shots = load_shots(config, dataset_type="mistakes", noise_model="bit_flip", agent_name="bp")
    env = QLDPCEnv(config, shots)
    agent = RouterAgent(config, env, router_checkpoint="checkpoints/evaluate_cps/router.pt")
    expert_list, inference_list = get_decoders(config, env)

    rl_train_loop(config, env, agent, expert_list, inference_list, train=False)


def train_moe_rl(config):
    shots = load_shots(config, dataset_type="moe")
    env = QLDPCEnv(config, shots)
    agent = RouterAgent(config, env, encoder_checkpoint="checkpoints/evaluate_cps/sl_nbp_big.pt")
    expert_list, inference_list = get_decoders(config, env)
    rl_train_loop(config, env, agent, expert_list, inference_list)


def train_router(config):
    train_moe_rl(config)
    # evaluate_moe(config)