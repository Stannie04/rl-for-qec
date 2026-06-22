import time
from tqdm import tqdm
import torch
from src.environment import QLDPCEnv
from src.agents import DQNAgent, SACAgent, MoEAgent, BPAgent, BPOSDAgent, NeuralBPPretrainer, NeuralBPEncoder
from src.train_utils import load_shots


def rl_train_loop(config, env, agent, expert_list, train=True):
    start_time = time.time()

    successes = 0
    losses = 0
    hits = 0
    action_dist = 0

    results = {}

    for step in tqdm(range(config.moe_num_timesteps)):
            # error_rate = get_physical_error_rate(config, step)
            # env.curriculum_error_rate = error_rate
            obs, info = env.reset()
            pattern = env.code.number_of_overlapping_stabilizers()
            results.setdefault(pattern, {"hits": 0, "successes": 0, "action_dist": 0})

            action, log_prob = agent.select_action(obs, evaluate=not train)

            # decoder = expert_list[action]
            decoder = expert_list[0] # For now, just use BP always

            reward = decoder_inference(config, decoder, env, obs, selected_decoder=action)

            if train:
                loss = agent.update(log_prob, reward)
                losses += loss

            results[pattern]["hits"] += 1
            results[pattern]["successes"] += 1 if reward > 0 else 0
            # results[pattern]["bp_successes"] += 1 if (reward > 0 and action == 0) or (reward < 0 and action != 0) else 0
            results[pattern]["action_dist"] += 1 if action == 0 else 0

            if train and step % 100 == 0:
                print(f"Step {step+1}/{config.moe_num_timesteps}, MOE success rate: {successes / 100}, BP success rate: {hits / 100}%, BP chosen: {action_dist / 100}%")
                losses = 0
                successes = 0
                action_dist = 0
                hits = 0

            # print(f"Step {step+1}/{config.moe_num_timesteps}, Action: {action}, Reward: {reward:.4f}, Loss: {loss:.4f}")

    # print(f"MOE success rate: {successes / config.moe_num_timesteps}, BP success rate: {hits / config.moe_num_timesteps}%, BP chosen: {action_dist / config.moe_num_timesteps}%")
    print(f"MoE Training completed in {time.time() - start_time:.2f} seconds. Success rate: {successes / config.moe_num_timesteps:.4f}")

    print("NOTE: BP success rate should be 0, since we are evaluating ability to predict BP success.")
    for pattern, pattern_results in results.items():
        total = pattern_results["hits"]
        success_rate = pattern_results["successes"] / total if total > 0 else 0
        action_dist_rate = pattern_results["action_dist"] / total if total > 0 else 0
        print(f"Pattern: {pattern}, Hits: {total}, MOE Success Rate: {success_rate:.4f}, Action distribution (BP chosen): {action_dist_rate:.4f}")


def decoder_inference(config, agent, env, obs, selected_decoder=None):

    start_time = time.time()
    done = False
    while not done:
        action, _ = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    end_time = time.time()

    bp_success = info["error_free"]
    bp_success = bp_success.item() if isinstance(bp_success, torch.Tensor) else bp_success
    prediction = (selected_decoder == 0)

    # raw_reward = error_free - config.moe_time_penalty_factor * (end_time - start_time)

    # Currently, we use the router to predict whether BP will succeed or not.
    return 1 if prediction == bp_success else -1


def get_decoders(config, env):
    expert_list = []

    for decoder in config.moe_experts:
        match decoder:
            case "bp": expert_list.append(BPAgent(env, config))
            case "bp_osd": expert_list.append(BPOSDAgent(env, config))
            case "neural_bp":
                expert_list.append(None)
                # encoder = NeuralBPEncoder(config, env)
                # encoder.load_state_dict(checkpoint["encoder"])
                # agent = NeuralBPPretrainer(encoder, config)
                # checkpoint = torch.load(f"checkpoints/encoder.pt", map_location=config.device)
                # agent.output_layer.load_state_dict(checkpoint["output_layer"])
                # agent.eval()
                # expert_list.append(agent)

            case "sac":
                agent = SACAgent(env, config)
                checkpoint = torch.load(f"checkpoints/{config.agent_name}_{config.code_name}.pt", map_location=config.device)
                agent.actor.load_state_dict(checkpoint["actor"])
                agent.critic1.load_state_dict(checkpoint["critic1"])
                agent.critic2.load_state_dict(checkpoint["critic2"])
                expert_list.append(agent)
            case _: raise ValueError(f"Unknown decoder: {decoder}")

    return expert_list


def evaluate_moe(config):
    shots = load_shots(config, dataset_type="mistakes", agent_name="bp")
    env = QLDPCEnv(config, shots)
    agent = MoEAgent(config, env)

    checkpoint = torch.load(f"checkpoints/router.pt", map_location=config.device)
    agent.router.load_state_dict(checkpoint)
    expert_list = get_decoders(config, env)

    rl_train_loop(config, env, agent, expert_list, train=False)


def train_moe_rl(config):
    shots = load_shots(config, dataset_type="moe")
    env = QLDPCEnv(config, shots)
    agent = MoEAgent(config, env)
    agent.router.encoder.load_state_dict(torch.load(f"checkpoints/encoder.pt", map_location=config.device)["encoder"])
    expert_list = get_decoders(config, env)
    # rl_train_loop(config, env, agent, expert_list)
    # agent.save()

def train_moe(config):
    # train_moe_rl(config)
    evaluate_moe(config)