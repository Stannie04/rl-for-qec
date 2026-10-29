from src.train_utils.datasets import load_shots, create_dataset_from_random_shots_labelled
from src.train_utils.inference import get_agent_and_inference, inference
from src.environment import QECEnv
import time
from tqdm import tqdm
import numpy as np

def evaluate_latency(config):
    for i in [5, 10, 20, 30, 50, 100]:
        test = np.load(f"datasets/144_12_12_ldpc/mistakes/depolarizing/random_labelled/bp_{i}_latencies.npy")
        print(f"Average latency: {np.mean(test)}")
        test = np.load(f"datasets/144_12_12_ldpc/mistakes/depolarizing/random_labelled/bp_osd_{i}_latencies.npy")
        print(f"Average latency: {np.mean(test)}")
        test = np.load(f"datasets/144_12_12_ldpc/mistakes/depolarizing/random_labelled/bp_lsd_{i}_latencies.npy")
        print(f"Average latency: {np.mean(test)}")
        print()