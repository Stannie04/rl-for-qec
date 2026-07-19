from collections import Counter

from src.environment import QLDPCEnv
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.train_utils.datasets import load_shots, create_dataset_from_all_permutations, create_dataset_from_random_shots
from src.train_utils.plotting import plot_jaccard_heatmap, plot_oracle_mistakes
from PIL import Image

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


def compute_overlap_stats(
    mistakes: np.ndarray,
    other_mistakes: np.ndarray,
    error_counts: np.ndarray,
    other_agent_name: str = "Other",
):

    if mistakes.ndim != 3 or other_mistakes.ndim != 3:
        raise ValueError("Expected arrays of shape (N, 2, D)")

    if mistakes.shape[1:] != other_mistakes.shape[1:]:
        raise ValueError("Inner shapes must match")

    if mistakes.dtype != other_mistakes.dtype:
        mistakes = mistakes.astype(np.int8)
        other_mistakes = other_mistakes.astype(np.int8)
        # raise ValueError(f"Dtypes must match: got {mistakes.dtype} and {other_mistakes.dtype}")

    # Make contiguous for safe view casting
    mistakes = np.ascontiguousarray(mistakes)
    other_mistakes = np.ascontiguousarray(other_mistakes)
    record_dtype = np.dtype((np.void, mistakes.dtype.itemsize * np.prod(mistakes.shape[1:])))
    mistakes_view = mistakes.reshape(mistakes.shape[0], -1).view(record_dtype).ravel()
    other_view = (other_mistakes.reshape(other_mistakes.shape[0], -1).view(record_dtype).ravel())

    # Vectorized membership check
    overlap_mask = np.isin(mistakes_view, other_view)

    overlap = overlap_mask.sum()

    print(
        f"\n{other_agent_name}: "
        f"{overlap}/{len(mistakes)} "
        f"({100 * overlap / len(mistakes):.2f}%)"
    )

    # Fast grouped stats
    max_err = int(error_counts.max())
    total_per_bin = np.bincount(error_counts, minlength=max_err + 1)
    overlap_per_bin = np.bincount(
        error_counts[overlap_mask],
        minlength=max_err + 1,
    )
    for n_errors in np.nonzero(total_per_bin)[0]:
        total = total_per_bin[n_errors]
        overlap_n = overlap_per_bin[n_errors]

        print(
            f"    {n_errors:2d} initial errors : "
            f"{overlap_n:4d}/{total:4d} "
            f"({100 * overlap_n / total:6.2f}%)"
        )

    return overlap_mask


def analyze_datasets(config, agents):
    all_mistakes = {}
    for agent_name in agents:
        mistakes = load_shots(
            config,
            dataset_type="mistakes",
            noise_model="bit_flip",
            agent_name=agent_name,
        )
        all_mistakes[agent_name] = mistakes

    compute_all_mistake_overlaps(config, all_mistakes)

    # agent_names, jaccard = compute_jaccard_matrix(all_mistakes)
    # plot_jaccard_heatmap(agent_names, jaccard)

    plot_oracle_mistakes(all_mistakes)


def compute_all_mistake_overlaps(config, all_mistakes):

    random_shots = load_shots(config, dataset_type="uniform")
    random_shot_distribution = {}
    for shot in tqdm(random_shots, leave=False):
        num_errors = shot[0].sum()
        random_shot_distribution[num_errors] = random_shot_distribution.get(num_errors, 0) + 1

    print("\nShot Distribution:")
    for num_errors, count in sorted(random_shot_distribution.items()):
        print(f"  {num_errors} errors: {count} samples")


    for agent_name, mistakes in all_mistakes.items():
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

        for other_agent_name, other_mistakes in all_mistakes.items():
            if other_agent_name == agent_name:
                continue

            # Overall overlap
            compute_overlap_stats(mistakes, other_mistakes, error_counts, other_agent_name=other_agent_name)


def compute_jaccard_matrix(all_mistakes):

    def mistakes_to_set(mistakes):
        mistakes = np.ascontiguousarray(mistakes)
        flat = mistakes.reshape(mistakes.shape[0], -1)
        return {row.tobytes() for row in flat}


    agent_names = list(all_mistakes.keys())
    mistake_sets = {
        name: mistakes_to_set(mistakes)
        for name, mistakes in all_mistakes.items()
    }

    n = len(agent_names)
    jaccard = np.zeros((n, n))

    for i, a in enumerate(agent_names):
        for j, b in enumerate(agent_names):

            inter = len(mistake_sets[a] & mistake_sets[b])
            union = len(mistake_sets[a] | mistake_sets[b])

            jaccard[i, j] = inter / union if union else 1.0

    return agent_names, jaccard


def get_nonzero_overlap_distribution(config, shots):

    env = QLDPCEnv(config, shots)

    H_z = env.code.H_z.cpu().numpy()
    error_matrix = shots[:, 0, :]

    overlaps = error_matrix @ H_z.T

    num_one = (overlaps == 1).sum(axis=1)
    num_two = (overlaps == 2).sum(axis=1)

    all_overlaps = list(zip(num_one.tolist(), num_two.tolist()))
    overlap_counts = Counter(all_overlaps)

    print("\nOverlap distribution for uniform shots:")
    for overlap, count in sorted(overlap_counts.items()):
        print(f"  Overlap {overlap}: {count} samples")

    return overlap_counts


def get_mistake_distribution(config, agents):
    shots = load_shots(config, dataset_type="uniform", noise_model="bit_flip")
    all_overlap_counts = get_nonzero_overlap_distribution(config, shots)

    # all_shots = create_dataset_from_all_permutations(config, num_errors=[1,2,3,4])
    # all_shot_counts = get_nonzero_overlap_distribution(config, all_shots)


    env_tmp = QLDPCEnv(config)
    H_z = env_tmp.code.H_z.detach().cpu().numpy()
    H_z_T = H_z.T

    all_accuracies = {}

    for agent_name in agents:
        shots = load_shots(config,dataset_type="mistakes",noise_model="bit_flip",agent_name=agent_name)

        error_matrix = np.asarray([shot[0] for shot in shots], dtype=np.int8)

        overlaps_mat = error_matrix @ H_z_T

        n_one = np.sum(overlaps_mat == 1, axis=1)
        n_two = np.sum(overlaps_mat == 2, axis=1)

        all_mistake_overlaps = list(zip(n_one, n_two))

        mistake_counts = Counter(all_mistake_overlaps)

        print("\nOverlap distribution for mistake shots:")
        for overlap, count in sorted(mistake_counts.items()):
            print(f"  Overlap {overlap}: {count} samples")

        representative_shots = {}
        seen = set()

        for shot, overlap in zip(shots, all_mistake_overlaps):
            if overlap not in seen:
                representative_shots[overlap] = shot
                seen.add(overlap)

        images = []
        images_untitled = []
        for overlap, shot in representative_shots.items():
            mistake_count = mistake_counts[overlap]
            all_overlaps_count = all_overlap_counts[overlap]

            env_tmp.reset_with_error_pattern(shot[0], shot[1])
            img = env_tmp.code.render_subgraph(
                np.array(np.where(shot[0] == 1)[0]),
                overlap,
                mistake_count,
                all_overlaps_count,
                with_title=True
            )
            images.append((overlap, img))

            img_untitled = env_tmp.code.render_subgraph(
                np.array(np.where(shot[0] == 1)[0]),
                overlap,
                mistake_count,
                all_overlaps_count,
                with_title=False
            )
            images_untitled.append((overlap, img_untitled))

        if not images:
            continue

        # save untitled images
        for overlap, img in images_untitled:
            img.save(f"results/patterns/{overlap}.png")

        images.sort(key=lambda item: (item[0][0], item[0][1]))
        image_size = images[0][1].size

        cols = 4
        rows = math.ceil(len(images) / cols)

        canvas = Image.new(
            "RGB",
            (cols * image_size[0], rows * image_size[1]),
            "white",
        )

        for i, (_, img) in enumerate(images):
            x = (i % cols) * image_size[0]
            y = (i // cols) * image_size[1]
            canvas.paste(img, (x, y))

        canvas.save(f"results/mistake_overlap_distribution_{agent_name}.png")
        # canvas.show()

        all_accuracies[agent_name] = {pattern: 1-(count / all_overlap_counts[pattern]) for pattern, count in mistake_counts.items()}

    # print all_accuracies as a table
    print("\nMistake accuracies by overlap pattern:")
    print("-" * 95)
    header = ["Overlap"] + list(all_accuracies.keys())
    header_format = " ".join(["{:<15}"] * len(header))
    print(header_format.format(*header))
    print("-" * 95)
    for pattern in sorted(all_overlap_counts.keys()):
        if pattern in [(3,3), (4,4), (5,2), (6,3), (8,2), (9,0), (10,1), (12,0)]:
            row = [f"{pattern}"]
            for agent_name in all_accuracies.keys():
                accuracy = all_accuracies[agent_name].get(pattern, 1.0)
                row.append(f"{accuracy:.3%}")
            print(header_format.format(*row))


def get_pattern_frequency(config):
    num_samples = int(1e7)

    shots = create_dataset_from_random_shots(config, num_samples=num_samples, error_rate=0.001)
    env = QLDPCEnv(config, shots)

    H_z = env.code.H_z.cpu().numpy()
    error_matrix = shots[:, 0, :]

    overlaps = error_matrix @ H_z.T

    num_one = (overlaps == 1).sum(axis=1)
    num_two = (overlaps == 2).sum(axis=1)

    all_overlaps = list(zip(num_one.tolist(), num_two.tolist()))
    overlap_counts = Counter(all_overlaps)

    num_errors = num_samples - overlap_counts[(0,0)]

    print("\nOverlap distribution for uniform shots:")
    irrelevant_sum = 0
    solvable_sum = 0
    for overlap, count in sorted(overlap_counts.items()):
        if overlap in [(3,3), (4,4), (5,2), (6,3), (8,2), (9,0), (10,1), (12,0)]:
            print(f"  Overlap {overlap}: {count} samples, Frequency: {count / num_samples:.8}, Errors: {count/num_errors:.6}")
        elif overlap in [(3,0), (4,1), (6,0), (7,1)]:
            solvable_sum += count
        elif overlap != (0,0):
            irrelevant_sum += count
    print(f"  Solvable: {solvable_sum} samples, Frequency: {solvable_sum / num_samples:.6%}, Errors: {solvable_sum/num_errors:.4%}")
    print(f"  Higher weight: {irrelevant_sum} samples, Frequency: {irrelevant_sum / num_samples:.6%}, Errors: {irrelevant_sum/num_errors:.4%}")


def get_absolute_error_rate(config):
    mistakes = load_shots(config, dataset_type="mistakes", noise_model="bit_flip", agent_name="bp")

    mistake_distribution = {}
    for shot in tqdm(mistakes, leave=False):
        num_errors = shot[0].sum()
        mistake_distribution[num_errors] = mistake_distribution.get(num_errors, 0) + 1

    print("\nShot Distribution:")
    for num_errors, count in sorted(mistake_distribution.items()):
        print(f"  {num_errors} errors: {count} samples")

    num_qubits = config.n
    for error_rate in [0.001, 0.0025, 0.005, 0.0075, 0.01]:

        depolarizing_error_rate = error_rate / 3

        total_probability = 0.0
        depolarizing_probability = 0.0

        success_rate = 1.0 - error_rate
        success_rate_depolarizing = 1.0 - depolarizing_error_rate

        for num_errors, num_samples in mistake_distribution.items():
            num_successes = num_qubits - num_errors
            total_probability += ((error_rate ** num_errors) * (success_rate ** num_successes)) * num_samples
            depolarizing_probability += (depolarizing_error_rate ** num_errors) * (success_rate_depolarizing ** num_successes) * num_samples

        print(f"p = {error_rate}: LER = {total_probability:e}, Depolarizing LER = {depolarizing_probability:e}")



def full_analysis(config):
    env = QLDPCEnv(config)
    code = env.code
    agents = ["bp", "sl_nbp_big", "sl_tanner_big", "sac_nbp_big",  "sac_tanner_big"]
    print(f"Code Name: {config.code_name}\n")
    # probabilities_of_k_errors_per_shot(code)
    # get_pattern_frequency(config)
    analyze_datasets(config, agents)
    # get_mistake_distribution(config, agents)
