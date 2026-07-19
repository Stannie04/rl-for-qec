import torch
from src.agents import SACAgent, BPAgent, BPOSDAgent, MWPMAgent, SLAgent
from src.environment import QLDPCEnv
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm


def get_worker_fn(task):
    match task:
        case "failures": return count_failures
        case "mistakes": return collect_mistakes
        case _: raise NotImplementedError(f"Task {task} is not implemented.")


def parallel_inference(agent_name, config, shots, task, repetition=None):

    shot_chunks = np.array_split(shots, config.num_workers)

    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        chunk_results = list(executor.map(
            run_worker,
            [agent_name]*config.num_workers,
            [config] * config.num_workers,
            shot_chunks,
            [get_worker_fn(task)] * config.num_workers,
            range(config.num_workers),
            [repetition] * config.num_workers))

    results = np.concatenate(chunk_results)

    # with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
    #     futures = [executor.submit(run_worker, agent_name, config, chunk, get_worker_fn(task), worker_id) for worker_id, chunk in enumerate(shot_chunks)]
    #     for future in as_completed(futures):
    #         results.append(future.result())

    return results


def count_failures(agent, env, inference_fn, worker_id):
    # failures = 0
    # for _ in tqdm(env.shots, desc=f"Worker {worker_id}", position=worker_id, leave=True):
    #     failures += not inference_fn(agent, env)
    # return failures

    failures = np.empty(len(env.shots), dtype=bool)
    for i, _ in enumerate(tqdm(env.shots, desc=f"Worker {worker_id}", position=worker_id, leave=True)):
        failures[i] = not inference_fn(agent, env)
    return failures


def collect_mistakes(agent, env, inference_fn, worker_id):
    mistakes = []
    for x, z in tqdm(env.shots, desc=f"Worker {worker_id}", position=worker_id, leave=True):
        if not inference_fn(agent, env):
            mistakes.append((x, z))
    return mistakes


def run_worker(agent_name, config, shots, worker_fn, worker_id, repetition=None):
    env = QLDPCEnv(config, shots)
    agent, inference_fn = get_agent_and_inference(config, env, agent_name, repetition)
    return worker_fn(agent, env, inference_fn, worker_id)


def get_agent_and_inference(config, env, agent_name, repetition=None):
    agent = get_agent(config, env, agent_name, repetition)
    inference_fn = None
    if type(agent_name) == str:
        if agent_name.startswith("bp") or agent_name.startswith("mwpm"):
            inference_fn = classical_inference
        elif agent_name.startswith("sac") or agent_name.startswith("rl"):
            inference_fn = rl_inference
        elif agent_name.startswith("sl"):
            inference_fn = sl_inference
        else:
            raise NotImplementedError(f"Inference function for agent {agent_name} is not implemented.")

    return agent, inference_fn


def get_agent(config, env, agent_name, repetition=None):
    if type(agent_name) == str:
        old = config.use_neural_bp, config.hidden_layers_gnn, config.hidden_layers_mlp, config.neural_bp_iterations
        match agent_name:

            case "rl_nbp":
                old = config.use_neural_bp, config.hidden_layers_gnn, config.hidden_layers_mlp, config.neural_bp_iterations
                config.use_neural_bp = True
                config.hidden_layers_gnn = [32,128,256]
                config.hidden_layers_mlp = [256,128,64]
                config.neural_bp_iterations = 4

                checkpoint = f"checkpoints/evaluate_cps/rl_nbp.pt" if repetition is None else f"checkpoints/repetition_cps/rl_nbp/rl_nbp_{repetition}.pt"
                agent = SACAgent(env, config, checkpoint=checkpoint)

            case "rl_cgnn":
                old = config.use_neural_bp, config.hidden_layers_gnn, config.hidden_layers_mlp, config.neural_bp_iterations
                config.hidden_layers_gnn = [32,128,256,512,512,256,256]
                config.hidden_layers_mlp = [256,128,64]
                config.use_neural_bp = False
                config.neural_bp_iterations = 8

                checkpoint = f"checkpoints/evaluate_cps/rl_cgnn.pt" if repetition is None else f"checkpoints/repetition_cps/rl_cgnn/rl_cgnn_{repetition}.pt"
                agent = SACAgent(env, config, checkpoint=checkpoint)

            case "sl_nbp":

                config.hidden_layers_gnn = [32,128,256,512,512,256,256]
                config.hidden_layers_mlp = [256,128,64]
                config.use_neural_bp = True
                config.neural_bp_iterations = 8
                checkpoint = f"checkpoints/evaluate_cps/sl_nbp.pt" if repetition is None else f"checkpoints/repetition_cps/sl_nbp/sl_nbp_{repetition}.pt"
                agent = SLAgent(config, env, nbp_checkpoint=checkpoint).to(config.device)
                agent.eval()

            case "sl_cgnn":
                old = config.use_neural_bp, config.hidden_layers_gnn, config.hidden_layers_mlp, config.neural_bp_iterations
                config.hidden_layers_gnn = [32,128,256,512,512,256,256]
                config.hidden_layers_mlp = [256,128,64]
                config.use_neural_bp = False
                checkpoint = f"checkpoints/evaluate_cps/sl_cgnn.pt" if repetition is None else f"checkpoints/repetition_cps/sl_cgnn/sl_cgnn_{repetition}.pt"
                agent = SLAgent(config, env, cgnn_checkpoint=checkpoint).to(config.device)
                agent.eval()

            case "mwpm": return MWPMAgent(env, config)
            case "bp": return BPAgent(env, config)
            case "bp_osd": return BPOSDAgent(env, config)
            case _: raise NotImplementedError(f"Agent {agent_name} does not exist.")

        config.use_neural_bp, config.hidden_layers_gnn, config.hidden_layers_mlp, config.neural_bp_iterations = old
        return agent
    else:
        return agent_name


def classical_inference(agent, env, state=None):

    obs, info = state if state is not None else env.reset()
    done = info["error_free"]

    while not done:
        error_pred, _ = agent.select_action(obs)

        if len(error_pred) == 0:
            break

        for a in error_pred:
            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

    return info["error_free"]


def rl_inference(agent, env, state=None):

    obs, info = state if state is not None else env.reset()
    done = info["error_free"]

    while not done:
        action, _ = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return info["error_free"]


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
