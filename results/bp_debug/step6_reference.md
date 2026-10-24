# Step 6: MWPM reference and logical-check cross-check

Dataset: iid depolarizing shots at p in {0.001, 0.003, 0.005} (`random_shots`, 10 shots per p) and uniform shots of nominal weight 1-3 (`uniform_shots`, 10 shots per weight; weight w means w X flips AND w Z flips at independent random positions, so up to 2w qubits are touched). Codes: d3_surface. Seed 0. Decoders: pymatching MWPM on the check matrices (`Matching(H_z)` for X errors, `Matching(H_x)` for Z), the repo `MWPMAgent` path (`classical_inference`, i.e. `has_logical_error`), and plain BP (`ldpc.BpDecoder`, prior 2*curriculum_error_rate/3, max_iter=30). MWPM and BP are scored with `common.independent_logical_failure`; `disagreements` counts shots where the repo verdict differs from that check on the same MWPM prediction.

## iid depolarizing shots

| code | p | n | MWPM fails | MWPM frac | BP fails | BP frac | BP non-conv | repo MWPM fails | disagreements | disagreement shot indices |
|---|---|---|---|---|---|---|---|---|---|---|
| d3_surface | 0.001 | 10 | 0 | 0.0000 | 0 | 0.0000 | 0 | 0 | 0 | - |
| d3_surface | 0.003 | 10 | 0 | 0.0000 | 0 | 0.0000 | 0 | 0 | 0 | - |
| d3_surface | 0.005 | 10 | 0 | 0.0000 | 0 | 0.0000 | 0 | 0 | 0 | - |

## Uniform shots, weights 1-3

| code | weight | n | MWPM fails | MWPM frac | BP fails | BP frac | BP non-conv | repo MWPM fails | disagreements | disagreement shot indices |
|---|---|---|---|---|---|---|---|---|---|---|
| d3_surface | 1 | 10 | 0 | 0.0000 | 4 | 0.4000 | 4 | 0 | 0 | - |
| d3_surface | 2 | 10 | 8 | 0.8000 | 10 | 1.0000 | 8 | 8 | 0 | - |
| d3_surface | 3 | 10 | 9 | 0.9000 | 10 | 1.0000 | 7 | 9 | 0 | - |

## Observation

- Repo `has_logical_error` evaluation and the independent logical check disagree on 0 of 60 MWPM predictions over all 6 datasets.
- On the uniform weight-1..3 shots MWPM fails 17 times and plain BP fails 24 times (30 shots in total).
- Weight-1 shots (one X and one Z flip at independent positions, 10 shots per code): MWPM fails 0 times, BP fails 4 times.
- [d3_surface] iid p=0.001 (n=10): MWPM fails 0, BP fails 0 (non-converged BP shots: 0).
