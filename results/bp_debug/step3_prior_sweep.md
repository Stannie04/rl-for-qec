# Step 3: BP prior sweep

Dataset: part A `create_dataset_from_uniform_shots` (nominal weight w = w X flips AND w Z flips at independent positions, so up to 2w of the n qubits are touched) and part B `create_dataset_from_random_shots` (iid depolarizing, per-qubit Pauli rate p); sample sizes 5000 shots per (code, weight) and 20000 shots per (code, p); seeds: --seed=0 with per-code base seed seed + 100*code_index and part B using base + p_index + 1; decoder `common.decode_bp` (ldpc.BpDecoder, max_iter=30, product_sum, parallel, prior passed through unchanged).

## Part A: uniform shots, weights 1..3

Cell = BP failure fraction. `matched 2p/3 = w/n`, `2p = 3w/n`, `p = 3w/(2n)` (see module docstring).

| code | w | nominal p | matched_2p_over_3 | 2p | p | repo | 0.01 | 0.05 | 0.1 | n shots |
|---|---|---|---|---|---|---|---|---|---|---|
| d7_surface | 1 | 0.03061 | 0.4368 | 0.4368 | 0.4368 | 0.4368 | 0.4368 | 0.4368 | 0.4368 | 5000 |
| d7_surface | 2 | 0.06122 | 0.7350 | 0.7402 | 0.7348 | 0.7448 | 0.7448 | 0.7394 | 0.7448 | 5000 |
| d7_surface | 3 | 0.09184 | 0.8988 | 0.9298 | 0.9044 | 0.9072 | 0.9068 | 0.8996 | 0.9070 | 5000 |
| 288_8_12_tile | 1 | 0.00521 | 0.1366 | 0.1366 | 0.1366 | 0.1366 | 0.1366 | 0.1366 | 0.1366 | 5000 |
| 288_8_12_tile | 2 | 0.01042 | 0.2452 | 0.2452 | 0.2452 | 0.2452 | 0.2452 | 0.2452 | 0.2462 | 5000 |
| 288_8_12_tile | 3 | 0.01562 | 0.3630 | 0.3626 | 0.3632 | 0.3638 | 0.3632 | 0.3622 | 0.3664 | 5000 |
| 144_12_12_ldpc | 1 | 0.01042 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5000 |
| 144_12_12_ldpc | 2 | 0.02083 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5000 |
| 144_12_12_ldpc | 3 | 0.03125 | 0.0032 | 0.0046 | 0.0032 | 0.0046 | 0.0046 | 0.0046 | 0.0026 | 5000 |

## Part B: iid depolarizing shots

Cell = BP failure fraction. `matched 2p/3 = 2p/3`, `2p` and `p` use the row's p.

| code | p | matched_2p_over_3 | 2p | p | repo | 0.01 | 0.05 | 0.1 | n shots |
|---|---|---|---|---|---|---|---|---|---|
| d7_surface | 0.001 | 0.0159 | 0.0159 | 0.0159 | 0.0159 | 0.0159 | 0.0159 | 0.0159 | 20000 |
| d7_surface | 0.002 | 0.0343 | 0.0343 | 0.0343 | 0.0343 | 0.0343 | 0.0343 | 0.0343 | 20000 |
| d7_surface | 0.003 | 0.0481 | 0.0481 | 0.0481 | 0.0481 | 0.0481 | 0.0481 | 0.0481 | 20000 |
| d7_surface | 0.004 | 0.0626 | 0.0626 | 0.0626 | 0.0626 | 0.0626 | 0.0624 | 0.0626 | 20000 |
| d7_surface | 0.005 | 0.0750 | 0.0750 | 0.0750 | 0.0750 | 0.0750 | 0.0748 | 0.0750 | 20000 |
| 288_8_12_tile | 0.001 | 0.0283 | 0.0283 | 0.0283 | 0.0283 | 0.0283 | 0.0283 | 0.0283 | 20000 |
| 288_8_12_tile | 0.002 | 0.0537 | 0.0537 | 0.0537 | 0.0537 | 0.0537 | 0.0537 | 0.0539 | 20000 |
| 288_8_12_tile | 0.003 | 0.0762 | 0.0762 | 0.0762 | 0.0762 | 0.0762 | 0.0762 | 0.0767 | 20000 |
| 288_8_12_tile | 0.004 | 0.0945 | 0.0945 | 0.0945 | 0.0945 | 0.0945 | 0.0945 | 0.0949 | 20000 |
| 288_8_12_tile | 0.005 | 0.1281 | 0.1281 | 0.1281 | 0.1282 | 0.1281 | 0.1281 | 0.1287 | 20000 |
| 144_12_12_ldpc | 0.001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20000 |
| 144_12_12_ldpc | 0.002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20000 |
| 144_12_12_ldpc | 0.003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20000 |
| 144_12_12_ldpc | 0.004 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0000 | 20000 |
| 144_12_12_ldpc | 0.005 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 20000 |

## Repo prior per code

| code | n | curriculum_error_rate | repo prior = 2*curriculum/3 |
|---|---|---|---|
| d7_surface | 49 | 0.00293878 | 0.00195918 |
| 288_8_12_tile | 288 | 0.0005 | 0.000333333 |
| 144_12_12_ldpc | 144 | 0.001 | 0.000666667 |

## Observation

- `d7_surface`: at iid p=0.003 the failure fraction spans 0.0481 (0.05) .. 0.0481 (matched_2p_over_3) over the 7 priors, repo prior 0.0481; at uniform weight 1 it spans 0.4368 .. 0.4368.
- `288_8_12_tile`: at iid p=0.003 the failure fraction spans 0.0762 (matched_2p_over_3) .. 0.0767 (0.1) over the 7 priors, repo prior 0.0762; at uniform weight 1 it spans 0.1366 .. 0.1366.
- `144_12_12_ldpc`: at iid p=0.003 the failure fraction spans 0.0000 (matched_2p_over_3) .. 0.0000 (matched_2p_over_3) over the 7 priors, repo prior 0.0000; at uniform weight 1 it spans 0.0000 .. 0.0000.
- Priors swept per row: matched 2p/3, 2p, p (from the row's nominal rate), the repo value 2*curriculum/3, and the fixed 0.01/0.05/0.1; 115000 shots per (code, prior).
