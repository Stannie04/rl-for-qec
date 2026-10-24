# Step 1: rerun baseline per weight

Dataset: `create_dataset_from_uniform_shots`, 5000 shots per nominal weight for weights 1..5 (25000 shots per code), seed 0; codes: d7_surface, 288_8_12_tile, 144_12_12_ldpc; full run. repo = `get_agent_and_inference` + `classical_inference`; common = `common.decode_bp` with `p_prior=2*curriculum_error_rate/3`, `max_iter=30`.

## Failure fraction (code x weight x decoder)

| code | weight | decoder | repo failures/n | repo failure fraction | common failures/n | common failure fraction | disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d7_surface | 1 | bp | 2184/5000 | 0.437 | 2184/5000 | 0.437 | 0 |
| d7_surface | 2 | bp | 3724/5000 | 0.745 | 3724/5000 | 0.745 | 0 |
| d7_surface | 3 | bp | 4536/5000 | 0.907 | 4536/5000 | 0.907 | 0 |
| d7_surface | 4 | bp | 4871/5000 | 0.974 | 4871/5000 | 0.974 | 0 |
| d7_surface | 5 | bp | 4953/5000 | 0.991 | 4953/5000 | 0.991 | 0 |
| d7_surface | 1 | bp_osd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| d7_surface | 2 | bp_osd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| d7_surface | 3 | bp_osd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| d7_surface | 4 | bp_osd | 234/5000 | 0.047 | 236/5000 | 0.047 | 2 |
| d7_surface | 5 | bp_osd | 771/5000 | 0.154 | 769/5000 | 0.154 | 2 |
| d7_surface | 1 | bp_lsd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| d7_surface | 2 | bp_lsd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| d7_surface | 3 | bp_lsd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| d7_surface | 4 | bp_lsd | 236/5000 | 0.047 | 236/5000 | 0.047 | 0 |
| d7_surface | 5 | bp_lsd | 769/5000 | 0.154 | 769/5000 | 0.154 | 0 |
| d7_surface | 1 | mwpm | 0/5000 | 0.000 | - | - | - |
| d7_surface | 2 | mwpm | 0/5000 | 0.000 | - | - | - |
| d7_surface | 3 | mwpm | 0/5000 | 0.000 | - | - | - |
| d7_surface | 4 | mwpm | 234/5000 | 0.047 | - | - | - |
| d7_surface | 5 | mwpm | 765/5000 | 0.153 | - | - | - |
| 288_8_12_tile | 1 | bp | 661/5000 | 0.132 | 661/5000 | 0.132 | 0 |
| 288_8_12_tile | 2 | bp | 1239/5000 | 0.248 | 1239/5000 | 0.248 | 0 |
| 288_8_12_tile | 3 | bp | 1778/5000 | 0.356 | 1778/5000 | 0.356 | 0 |
| 288_8_12_tile | 4 | bp | 2242/5000 | 0.448 | 2242/5000 | 0.448 | 0 |
| 288_8_12_tile | 5 | bp | 2645/5000 | 0.529 | 2645/5000 | 0.529 | 0 |
| 288_8_12_tile | 1 | bp_osd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| 288_8_12_tile | 2 | bp_osd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| 288_8_12_tile | 3 | bp_osd | 2/5000 | 0.000 | 2/5000 | 0.000 | 0 |
| 288_8_12_tile | 4 | bp_osd | 2/5000 | 0.000 | 1/5000 | 0.000 | 1 |
| 288_8_12_tile | 5 | bp_osd | 10/5000 | 0.002 | 11/5000 | 0.002 | 3 |
| 288_8_12_tile | 1 | bp_lsd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| 288_8_12_tile | 2 | bp_lsd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| 288_8_12_tile | 3 | bp_lsd | 1/5000 | 0.000 | 1/5000 | 0.000 | 0 |
| 288_8_12_tile | 4 | bp_lsd | 2/5000 | 0.000 | 2/5000 | 0.000 | 0 |
| 288_8_12_tile | 5 | bp_lsd | 8/5000 | 0.002 | 8/5000 | 0.002 | 0 |
| 288_8_12_tile | - | mwpm | error | - | - | - | - |
| 144_12_12_ldpc | 1 | bp | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| 144_12_12_ldpc | 2 | bp | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| 144_12_12_ldpc | 3 | bp | 30/5000 | 0.006 | 30/5000 | 0.006 | 0 |
| 144_12_12_ldpc | 4 | bp | 86/5000 | 0.017 | 86/5000 | 0.017 | 0 |
| 144_12_12_ldpc | 5 | bp | 162/5000 | 0.032 | 162/5000 | 0.032 | 0 |
| 144_12_12_ldpc | 1 | bp_osd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| 144_12_12_ldpc | 2 | bp_osd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| 144_12_12_ldpc | 3 | bp_osd | 26/5000 | 0.005 | 26/5000 | 0.005 | 0 |
| 144_12_12_ldpc | 4 | bp_osd | 79/5000 | 0.016 | 77/5000 | 0.015 | 2 |
| 144_12_12_ldpc | 5 | bp_osd | 137/5000 | 0.027 | 128/5000 | 0.026 | 21 |
| 144_12_12_ldpc | 1 | bp_lsd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| 144_12_12_ldpc | 2 | bp_lsd | 0/5000 | 0.000 | 0/5000 | 0.000 | 0 |
| 144_12_12_ldpc | 3 | bp_lsd | 24/5000 | 0.005 | 24/5000 | 0.005 | 0 |
| 144_12_12_ldpc | 4 | bp_lsd | 69/5000 | 0.014 | 69/5000 | 0.014 | 0 |
| 144_12_12_ldpc | 5 | bp_lsd | 119/5000 | 0.024 | 118/5000 | 0.024 | 1 |
| 144_12_12_ldpc | - | mwpm | error | - | - | - | - |

## Pipeline agreement (repo vs common, per shot)

| code | decoder | disagreements | n |
| --- | --- | --- | --- |
| d7_surface | bp | 0 | 25000 |
| d7_surface | bp_osd | 4 | 25000 |
| d7_surface | bp_lsd | 0 | 25000 |
| 288_8_12_tile | bp | 0 | 25000 |
| 288_8_12_tile | bp_osd | 4 | 25000 |
| 288_8_12_tile | bp_lsd | 0 | 25000 |
| 144_12_12_ldpc | bp | 0 | 25000 |
| 144_12_12_ldpc | bp_osd | 23 | 25000 |
| 144_12_12_ldpc | bp_lsd | 1 | 25000 |

## Observation

- Dataset: `create_dataset_from_uniform_shots` puts w X flips AND w Z flips at independent random positions, so a weight-w shot touches up to 2w of the n_data qubits; the nominal weight label comes from the weight-major row layout, not from the shot.
- d7_surface (n_data=49, p_prior=1.96e-03): repo failure fraction by weight 1..5 - bp 0.437/0.745/0.907/0.974/0.991, bp_osd 0.000/0.000/0.000/0.047/0.154, bp_lsd 0.000/0.000/0.000/0.047/0.154, mwpm 0.000/0.000/0.000/0.047/0.153.
- 288_8_12_tile (n_data=288, p_prior=3.33e-04): repo failure fraction by weight 1..5 - bp 0.132/0.248/0.356/0.448/0.529, bp_osd 0.000/0.000/0.000/0.000/0.002, bp_lsd 0.000/0.000/0.000/0.000/0.002. Repo path not constructible: mwpm unavailable (ValueError: `check_matrix` must contain at most two ones per column, but column 2 has 3 ones.).
- 144_12_12_ldpc (n_data=144, p_prior=6.67e-04): repo failure fraction by weight 1..5 - bp 0.000/0.000/0.006/0.017/0.032, bp_osd 0.000/0.000/0.005/0.016/0.027, bp_lsd 0.000/0.000/0.005/0.014/0.024. Repo path not constructible: mwpm unavailable (ValueError: `check_matrix` must contain at most two ones per column, but column 0 has 3 ones.).
- Repo vs common per-shot disagreements: d7_surface bp 0/25000; d7_surface bp_osd 4/25000; d7_surface bp_lsd 0/25000; 288_8_12_tile bp 0/25000; 288_8_12_tile bp_osd 4/25000; 288_8_12_tile bp_lsd 0/25000; 144_12_12_ldpc bp 0/25000; 144_12_12_ldpc bp_osd 23/25000; 144_12_12_ldpc bp_lsd 1/25000.
