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

def get_nonzero_overlap_distribution(config):
    shots = load_shots(config, dataset_type="uniform", noise_model="bit_flip")
    env = QLDPCEnv(config, shots)
    all_overlaps = []
    for shot in tqdm(shots, desc="Analyzing uniform shots", leave=False):
        env.reset_with_error_pattern(shot[0], shot[1])
        overlap = env.code.number_of_overlapping_stabilizers(np.array(np.where(shot[0] == 1)[0]))
        all_overlaps.append(overlap)

    overlap_counts = Counter(all_overlaps)

    print("\nOverlap distribution for uniform shots:")
    for overlap, count in sorted(overlap_counts.items()):
        print(f"  Overlap {overlap}: {count} samples")

    return overlap_counts


def get_mistake_distribution(config):

    # First, get the distribution of overlaps for the shot dataset that the mistake set is based on
    all_overlap_counts = get_nonzero_overlap_distribution(config)

    shots = load_shots(config, dataset_type="mistakes", noise_model="bit_flip")
    env = QLDPCEnv(config, shots)

    all_mistake_overlaps = []
    representative_shots = {}
    for shot in tqdm(shots, desc="Updating graph for mistake shots", leave=False):
        env.reset_with_error_pattern(shot[0], shot[1])
        overlap = env.code.number_of_overlapping_stabilizers(np.array(np.where(shot[0] == 1)[0]))

        if overlap not in all_mistake_overlaps:
            print(f"New overlap value: {overlap}")
            representative_shots[overlap] = shot

        all_mistake_overlaps.append(overlap)


    print("\nOverlap distribution for mistake shots:")
    mistake_counts = Counter(all_mistake_overlaps)
    for overlap, count in sorted(mistake_counts.items()):
        print(f"  Overlap {overlap}: {count} samples")


    for overlap, shot in representative_shots.items():
        mistake_count = mistake_counts[overlap]
        all_overlaps_count = all_overlap_counts[overlap]

        env.reset_with_error_pattern(shot[0], shot[1])
        env.code.render_subgraph(np.array(np.where(shot[0] == 1)[0]), overlap, mistake_count, all_overlaps_count)


    # Combine overlap graphs in directory results/overlap into a single plot
    images = []
    for overlap in sorted(mistake_counts.keys()):
        try:
            img = Image.open(f"results/overlap/{overlap}.png")
            images.append((overlap, img))
        except FileNotFoundError:
            print(f"Image for overlap {overlap} not found, skipping.")

    image_size = images[0][1].size

    cols, rows = 4, math.ceil(len(images) / 4)
    canvas = Image.new("RGB", (cols * image_size[0], rows * image_size[1]), "white")

    for i, img in enumerate(images):
        x = (i % cols) * image_size[0]
        y = (i // cols) * image_size[1]
        canvas.paste(img[1], (x, y))

    canvas.save("results/overlap/mistake_overlap_distribution.png")
    canvas.show()

def full_analysis(config):
    env = QLDPCEnv(config)
    code = env.code
    print(f"Code Name: {config.code_name}\n")
    # probabilities_of_k_errors_per_shot(code)
    # analyze_datasets(config)
    get_mistake_distribution(config)
