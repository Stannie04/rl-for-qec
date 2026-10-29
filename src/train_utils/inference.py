import torch
from src.agents import SACAgent, BPAgent, BPOSDAgent, BPLSDAgent, MWPMAgent, SLAgent, RouterNetwork
from src.environment import QECEnv
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import time

def get_worker_fn(task):
    match task:
        case "failures": return count_failures
        case "mistakes": return collect_mistakes
        case _: raise NotImplementedError(f"Task {task} is not implemented.")


def inference(agent_name, config, shots, task, repetition=None, parallel=True, checkpoint=None):
    worker_fn = get_worker_fn(task)

    if not parallel or config.num_workers == 1:
        result = run_worker(agent_name, config, shots, worker_fn, worker_id=0, repetition=repetition, checkpoint=checkpoint)
        return result if task == "mistakes" else np.array(result)

    shot_chunks = np.array_split(shots, config.num_workers)

    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        chunk_results = list(executor.map(
            run_worker,
            [agent_name]*config.num_workers,
            [config] * config.num_workers,
            shot_chunks,
            [worker_fn] * config.num_workers,
            range(config.num_workers),
            [repetition] * config.num_workers,
            [checkpoint] * config.num_workers,
        ))

    if task == "mistakes":
        return [mistake for chunk_mistakes, _ in chunk_results for mistake in chunk_mistakes], [latency for _, chunk_latencies in chunk_results for latency in chunk_latencies]

    return np.concatenate(chunk_results)


def count_failures(agent, env, inference_fn, worker_id):
    failures = np.empty(len(env.shots), dtype=bool)
    for i, _ in enumerate(tqdm(env.shots, desc=f"Worker {worker_id}", position=worker_id, leave=False)):
        failures[i] = not inference_fn(agent, env)
    return failures


def collect_mistakes(agent, env, inference_fn, worker_id):
    mistakes, latencies = [], []
    for x, z in tqdm(env.shots, desc=f"Worker {worker_id}", position=worker_id, leave=False):
        success, latency = inference_fn(agent, env)
        if not success:
            mistakes.append((x, z))
        latencies.append(latency)
    return mistakes, latencies


def run_worker(agent_name, config, shots, worker_fn, worker_id, repetition=None, checkpoint=None):
    env = QECEnv(config, shots)
    agent, inference_fn = get_agent_and_inference(config, env, agent_name, repetition, checkpoint)
    return worker_fn(agent, env, inference_fn, worker_id)


def get_agent_and_inference(config, env, agent_name, repetition=None, checkpoint=None):
    agent = get_agent(config, env, agent_name, repetition, checkpoint=checkpoint)
    inference_fn = rl_inference # Default inference function for RL agents, and returned if the agent is already an instance of an agent class.
    if type(agent_name) == str:
        if agent_name.startswith("bp") or agent_name.startswith("mwpm"):
            inference_fn = classical_inference
        elif agent_name.startswith("router"):
            inference_fn = router_inference
        elif agent_name.startswith("sac") or agent_name.startswith("rl"):
            inference_fn = rl_inference
        elif agent_name.startswith("sl"):
            inference_fn = sl_inference
        else:
            raise NotImplementedError(f"Inference function for agent {agent_name} is not implemented.")

    return agent, inference_fn


def get_agent(config, env, agent_name, repetition=None, checkpoint=None):
    if type(agent_name) == str:
        old = config.use_neural_bp, config.hidden_layers_cgnn, config.hidden_layers_mlp, config.neural_bp_iterations, config.neural_bp_hidden_dim
        checkpoint = checkpoint if checkpoint is not None else f"checkpoints/evaluate_cps/{agent_name}_{config.code_name}.pt" if repetition is None else f"checkpoints/repetition_cps/{agent_name}/{agent_name}_{repetition}.pt"
        match agent_name:

            case "rl_hpo": return SACAgent(env, config, checkpoint=checkpoint)

            case "router":
                checkpoint = f"checkpoints/evaluate_cps/router_{config.code_name}.pt" if checkpoint is None else checkpoint
                router = RouterNetwork(config, env, checkpoint)
                expert_list, inference_list = [], []
                for decoder_name in config.moe_experts:
                    decoder, inference = get_agent_and_inference(config, env, decoder_name)
                    expert_list.append(decoder)
                    inference_list.append(inference)
                agent = (router, expert_list, inference_list)

            case "rl_nbp":
                config.use_neural_bp = True
                config.hidden_layers_mlp = [256,128,64]
                config.neural_bp_iterations = 4
                config.neural_bp_hidden_dim = 64

                agent = SACAgent(env, config, checkpoint=checkpoint)

            case "rl_cgnn":
                config.hidden_layers_cgnn = [32,128,256,512,512,256,256]
                config.hidden_layers_mlp = [256,128,64]
                config.use_neural_bp = False
                config.neural_bp_iterations = 8
                config.neural_bp_hidden_dim = 128

                agent = SACAgent(env, config, checkpoint=checkpoint)

            case "sl_nbp":
                config.hidden_layers_mlp = [256,128,64]
                config.use_neural_bp = True
                config.neural_bp_iterations = 8
                config.neural_bp_hidden_dim = 128
                agent = SLAgent(config, env, nbp_checkpoint=checkpoint).to(config.device)
                agent.eval()

            case "sl_cgnn":
                config.hidden_layers_cgnn = [32,128,256,512,512,256,256]
                config.hidden_layers_mlp = [256,128,64]
                config.use_neural_bp = False
                agent = SLAgent(config, env, cgnn_checkpoint=checkpoint).to(config.device)
                agent.eval()

            case "mwpm": return MWPMAgent(env, config)
            case "bp": return BPAgent(env, config)
            case "bp_osd": return BPOSDAgent(env, config)
            case "bp_lsd": return BPLSDAgent(env, config)
            case _: raise NotImplementedError(f"Agent {agent_name} does not exist.")

        config.use_neural_bp, config.hidden_layers_cgnn, config.hidden_layers_mlp, config.neural_bp_iterations, config.neural_bp_hidden_dim = old
        return agent
    else:
        return agent_name


def classical_inference(agent, env, state=None):

    obs, info = state if state is not None else env.reset()

    error_pred, time = agent.select_action(obs)

    if len(error_pred) == 0:
        return env.code.is_syndrome_free() and not env.code.has_logical_error(), time

    for a in error_pred:
        env.step(a)

    return env.code.is_syndrome_free() and not env.code.has_logical_error(), time


def rl_inference(agent, env, state=None):

    obs, info = state if state is not None else env.reset()
    done = info["syndrome_free"]
    total_time = 0
    while not done:
        action, _, time = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_time += time
    return env.code.is_syndrome_free() and not env.code.has_logical_error(), total_time


def sl_inference(agent, env, state=None):

    obs, info = state if state is not None else env.reset()

    error_true = env.code.x_errors.float()
    with torch.no_grad():
        error_pred = agent(obs)

    true_indices = torch.nonzero(error_true).flatten()
    predicted_indices = torch.nonzero(error_pred > 0.5).flatten()

    pred_sorted = torch.sort(predicted_indices).values
    true_sorted = torch.sort(true_indices).values

    return torch.equal(pred_sorted, true_sorted)


def router_inference(agent, env, state=None):
    """NOTE: @ref agent consists of a router network, as well as a list of its experts."""

    router, expert_list, inference_list = agent

    obs, info = state if state is not None else env.reset()

    if info["syndrome_free"]:
        return True

    with torch.no_grad():
        pred, time = router(obs)
        action = int(pred.item() > 0)

    return inference_list[action](expert_list[action], env, state=(obs, info)), time