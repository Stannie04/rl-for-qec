import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np

def plot_results(results):

    plt.figure(figsize=(12, 6))
    for name, runs in results.items():
        mean, ci95 = get_confidence_bounds(runs)
        plt.plot(mean, label=f"{name} Mean", linewidth=2)
        plt.fill_between(range(len(mean)), mean - ci95, mean + ci95, alpha=0.3, label=f"{name} 95% CI")

    plt.title("RL Agent Performance on Multivariate Bicycle Code Environment")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.savefig("results/results.png")


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)


def get_confidence_bounds(results, window=20):
    """
    Calculates smoothed mean and 95% confidence intervals for the results.
    """
    mean = smooth(np.mean(results, axis=0), window=window)
    ci95 = 1.96 * np.std(results, axis=0) / np.sqrt(len(results))
    return mean, smooth(ci95, window=window)