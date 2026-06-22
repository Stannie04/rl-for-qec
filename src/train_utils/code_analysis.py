from collections import Counter

from src.environment import QLDPCEnv
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.train_utils.datasets import load_shots
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
    record_dtype = np.dtype(
        (np.void, mistakes.dtype.itemsize * np.prod(mistakes.shape[1:]))
    )
    mistakes_view = mistakes.reshape(mistakes.shape[0], -1).view(record_dtype).ravel()
    other_view = (
        other_mistakes.reshape(other_mistakes.shape[0], -1)
        .view(record_dtype)
        .ravel()
    )
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
    neural_bp_mistakes = np.load(f"datasets/{config.code_name}/mistakes_sac_finetuned_bit_flip.npy", allow_pickle=True)
    encoder_mistakes = np.load(f"datasets/{config.code_name}/mistakes_pretrained_encoder_bit_flip.npy", allow_pickle=True)
    finetuned_encoder_mistakes = np.load(f"datasets/{config.code_name}/mistakes_finetuned_encoder_bit_flip.npy", allow_pickle=True)
    bp_mistakes = np.load(f"datasets/{config.code_name}/mistakes_bp_bit_flip.npy", allow_pickle=True)
    bp_osd_mistakes = np.load(f"datasets/{config.code_name}/mistakes_bp_osd_bit_flip.npy", allow_pickle=True)

    for agent_name, mistakes in [
        ("SAC", sac_mistakes),
        # ("Neural BP", neural_bp_mistakes),
        ("BP", bp_mistakes),
        ("BP+OSD", bp_osd_mistakes),
        ("Encoder", encoder_mistakes),
        ("Finetuned Encoder", finetuned_encoder_mistakes),
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
            # ("Neural BP", neural_bp_mistakes),
            ("BP", bp_mistakes),
            ("BP+OSD", bp_osd_mistakes),
            ("Encoder", encoder_mistakes),
            ("Finetuned Encoder", finetuned_encoder_mistakes),
        ]:
            if other_agent_name == agent_name:
                continue

            # Overall overlap
            compute_overlap_stats(mistakes, other_mistakes, error_counts, other_agent_name=other_agent_name)


def get_nonzero_overlap_distribution(config):
    shots = load_shots(config, dataset_type="all", noise_model="bit_flip")
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


def get_mistake_distribution(config):

    all_overlap_counts = get_nonzero_overlap_distribution(config)

    env_tmp = QLDPCEnv(config)
    H_z = env_tmp.code.H_z.detach().cpu().numpy()
    H_z_T = H_z.T

    for agent_name in config.moe_experts:

        shots = load_shots(
            config,
            dataset_type="mistakes",
            noise_model="bit_flip",
            agent_name=agent_name,
        )

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
        for overlap, shot in representative_shots.items():
            mistake_count = mistake_counts[overlap]
            all_overlaps_count = all_overlap_counts[overlap]

            env_tmp.reset_with_error_pattern(shot[0], shot[1])
            img = env_tmp.code.render_subgraph(
                np.array(np.where(shot[0] == 1)[0]),
                overlap,
                mistake_count,
                all_overlaps_count,
            )
            images.append((overlap, img))

        if not images:
            continue

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
        canvas.show()


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
    for error_rate in [0.005, 0.01, 0.03, 0.05]:

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
    print(f"Code Name: {config.code_name}\n")
    # probabilities_of_k_errors_per_shot(code)
    # analyze_datasets(config)
    # get_mistake_distribution(config)
    get_absolute_error_rate(config)