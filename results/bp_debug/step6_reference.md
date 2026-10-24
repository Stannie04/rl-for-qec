# Step 6: MWPM reference and logical-check cross-check

Dataset: iid depolarizing shots at p in {0.001, 0.003, 0.005} (`random_shots`, 5000 shots per p) and uniform shots of nominal weight 1-3 (`uniform_shots`, 5000 shots per weight; weight w means w X flips AND w Z flips at independent random positions, so up to 2w qubits are touched). Codes: d7_surface, d5_surface. Seed 0. Decoders: pymatching MWPM on the check matrices (`Matching(H_z)` for X errors, `Matching(H_x)` for Z), the repo `MWPMAgent` path (`classical_inference`, i.e. `has_logical_error`), and plain BP (`ldpc.BpDecoder`, prior 2*curriculum_error_rate/3, max_iter=30). MWPM and BP are scored with `common.independent_logical_failure`; `disagreements` counts shots where the repo verdict differs from that check on the same MWPM prediction.

## iid depolarizing shots

| code | p | n | MWPM fails | MWPM frac | BP fails | BP frac | BP non-conv | repo MWPM fails | disagreements | disagreement shot indices |
|---|---|---|---|---|---|---|---|---|---|---|
| d7_surface | 0.001 | 5000 | 0 | 0.0000 | 81 | 0.0162 | 81 | 0 | 0 | - |
| d7_surface | 0.003 | 5000 | 0 | 0.0000 | 242 | 0.0484 | 242 | 0 | 0 | - |
| d7_surface | 0.005 | 5000 | 0 | 0.0000 | 378 | 0.0756 | 378 | 0 | 0 | - |
| d5_surface | 0.001 | 5000 | 0 | 0.0000 | 54 | 0.0108 | 54 | 0 | 0 | - |
| d5_surface | 0.003 | 5000 | 0 | 0.0000 | 182 | 0.0364 | 182 | 0 | 0 | - |
| d5_surface | 0.005 | 5000 | 0 | 0.0000 | 266 | 0.0532 | 266 | 0 | 0 | - |

## Uniform shots, weights 1-3

| code | weight | n | MWPM fails | MWPM frac | BP fails | BP frac | BP non-conv | repo MWPM fails | disagreements | disagreement shot indices |
|---|---|---|---|---|---|---|---|---|---|---|
| d7_surface | 1 | 5000 | 0 | 0.0000 | 2184 | 0.4368 | 2184 | 0 | 0 | - |
| d7_surface | 2 | 5000 | 0 | 0.0000 | 3724 | 0.7448 | 3724 | 0 | 0 | - |
| d7_surface | 3 | 5000 | 0 | 0.0000 | 4536 | 0.9072 | 4536 | 0 | 0 | - |
| d5_surface | 1 | 5000 | 0 | 0.0000 | 2708 | 0.5416 | 2708 | 0 | 0 | - |
| d5_surface | 2 | 5000 | 0 | 0.0000 | 4349 | 0.8698 | 4349 | 0 | 0 | - |
| d5_surface | 3 | 5000 | 1124 | 0.2248 | 4797 | 0.9594 | 4670 | 1124 | 0 | - |

## Observation

- Repo `has_logical_error` evaluation and the independent logical check disagree on 0 of 60000 MWPM predictions over all 12 datasets.
- On the uniform weight-1..3 shots MWPM fails 1124 times and plain BP fails 22298 times (30000 shots in total).
- Weight-1 shots (one X and one Z flip at independent positions, 5000 shots per code): MWPM fails 0 times, BP fails 4892 times.
- [d7_surface] iid p=0.001 (n=5000): MWPM fails 0, BP fails 81 (non-converged BP shots: 81).
- [d5_surface] iid p=0.001 (n=5000): MWPM fails 0, BP fails 54 (non-converged BP shots: 54).
