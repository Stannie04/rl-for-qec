from src.environment import QLDPCEnv
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.train_utils.datasets import load_shots

def probabilities_of_k_errors_per_shot(code):
    total_number_qubits = code.n_data
    error_probabilities = [0.001, 0.002, 0.003, 0.004, 0.005]

    rows = []
    for error_rate in error_probabilities:
        row = {"Error Rate": f"{error_rate:.3%}"}

        for k in range(6):
            prob_k_errors = (
                math.comb(total_number_qubits, k)
                * (error_rate ** k)
                * ((1 - error_rate) ** (total_number_qubits - k))
            )
            row[f"P_{k}"] = prob_k_errors
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", "{:.3e}".format)
    print(df.to_string(index=False))


def analyze_datasets(config):

    random_shots = load_shots(config, dataset_type="uniform")
    random_shot_distribution = {}
    for shot in tqdm(random_shots, leave=False):
        num_errors = shot[0].sum()
        random_shot_distribution[num_errors] = random_shot_distribution.get(num_errors, 0) + 1

    print("\nShot Distribution:")
    for num_errors, count in sorted(random_shot_distribution.items()):
        print(f"  {num_errors} errors: {count} samples")

    sac_mistakes = np.load(f"datasets/{config.code_name}/mistakes_sac_bit_flip.npy", allow_pickle=True)
    bp_mistakes = np.load(f"datasets/{config.code_name}/mistakes_bp_bit_flip.npy", allow_pickle=True)
    bp_osd_mistakes = np.load(f"datasets/{config.code_name}/mistakes_bp_osd_bit_flip.npy", allow_pickle=True)

    for agent_name, mistakes in [
        ("SAC", sac_mistakes),
        ("BP", bp_mistakes),
        ("BP+OSD", bp_osd_mistakes),
    ]:
        print("\n" + "=" * 80)
        print(f"{agent_name:^80}")
        print("=" * 80)

        if len(mistakes) == 0:
            print("No mistakes found.")
            continue

        error_counts = mistakes[:, 0, :].sum(axis=-1)
        values, counts = np.unique(error_counts, return_counts=True)

        print("\nMistake distribution:")
        print("-" * 80)
        for n_errors, n_mistakes in zip(values, counts):
            total_samples = random_shot_distribution[n_errors]
            pct = 100 * n_mistakes / total_samples
            print(
                f"  {n_errors:2d} initial errors : "
                f"{n_mistakes:5d} / {total_samples:5d} "
                f"({pct:6.2f}%)"
            )

        print("-" * 80)
        print(f"Total mistakes: {len(mistakes)}")

        print("\nOverlap analysis:")
        print("-" * 80)

        for other_agent_name, other_mistakes in [
            ("SAC", sac_mistakes),
            ("BP", bp_mistakes),
            ("BP+OSD", bp_osd_mistakes),
        ]:
            if other_agent_name == agent_name:
                continue

            # Overall overlap
            overlap_mask = np.array([
                any(np.array_equal(m, om) for om in other_mistakes)
                for m in mistakes
            ])
            overlap = overlap_mask.sum()

            print(
                f"\n{other_agent_name}: "
                f"{overlap}/{len(mistakes)} "
                f"({100 * overlap / len(mistakes):.2f}%)"
            )

            # Overlap broken down by initial error count
            for n_errors in np.unique(error_counts):
                subset = error_counts == n_errors
                n_subset = subset.sum()

                n_overlap = overlap_mask[subset].sum()

                print(
                    f"    {n_errors:2d} initial errors : "
                    f"{n_overlap:4d}/{n_subset:4d} "
                    f"({100 * n_overlap / n_subset:6.2f}%)"
                )


    # for agent_name, mistakes in [("SAC", sac_mistakes), ("BP", bp_mistakes), ("BP+OSD", bp_osd_mistakes)]:
    #     if len(mistakes) == 0:
    #         print(f"\n\n{agent_name} made no mistakes, skipping analysis.")
    #         continue
    #
    #     values, counts = np.unique(mistakes[:, 0, :].sum(axis=-1), return_counts=True)
    #     print(f"\n\n{agent_name} Mistakes Distribution:")
    #     for v, c in zip(values, counts):
    #         print(f"\t{v} errors: {c} mistakes in {random_shot_distribution[v]} samples ({c/random_shot_distribution[v]*100:.2f}%)")
    #     print(f"\nTotal samples: {len(mistakes)}")
    #     # Check if the other agents make the same mistakes
    #     for other_agent_name, other_mistakes in [("SAC", sac_mistakes), ("BP", bp_mistakes), ("BP+OSD", bp_osd_mistakes)]:
    #         if other_agent_name == agent_name:
    #             continue
    #         overlap = sum(any(np.array_equal(m, om) for om in other_mistakes) for m in mistakes)
    #         print(f"\tOverlap with {other_agent_name}: {overlap} samples ({overlap/len(mistakes)*100:.2f}%)")


def full_analysis(config):
    env = QLDPCEnv(config)
    code = env.code
    print(f"Code Name: {config.code_name}\n")
    probabilities_of_k_errors_per_shot(code)
    analyze_datasets(config)