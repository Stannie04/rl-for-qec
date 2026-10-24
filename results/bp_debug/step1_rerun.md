# Step 1: rerun baseline per weight

Dataset: `create_dataset_from_uniform_shots`, 5 shots per nominal weight for weights 1..5 (25 shots per code), seed 0; codes: d5_surface; quick (tiny) run. repo = `get_agent_and_inference` + `classical_inference`; common = `common.decode_bp` with `p_prior=2*curriculum_error_rate/3`, `max_iter=30`.

## Failure fraction (code x weight x decoder)

| code | weight | decoder | repo failures/n | repo failure fraction | common failures/n | common failure fraction | disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d5_surface | 1 | bp | 4/5 | 0.800 | 4/5 | 0.800 | 0 |
| d5_surface | 2 | bp | 5/5 | 1.000 | 5/5 | 1.000 | 0 |
| d5_surface | 3 | bp | 5/5 | 1.000 | 5/5 | 1.000 | 0 |
| d5_surface | 4 | bp | 5/5 | 1.000 | 5/5 | 1.000 | 0 |
| d5_surface | 5 | bp | 5/5 | 1.000 | 5/5 | 1.000 | 0 |
| d5_surface | 1 | bp_osd | 0/5 | 0.000 | 0/5 | 0.000 | 0 |
| d5_surface | 2 | bp_osd | 0/5 | 0.000 | 0/5 | 0.000 | 0 |
| d5_surface | 3 | bp_osd | 1/5 | 0.200 | 1/5 | 0.200 | 0 |
| d5_surface | 4 | bp_osd | 4/5 | 0.800 | 4/5 | 0.800 | 0 |
| d5_surface | 5 | bp_osd | 4/5 | 0.800 | 4/5 | 0.800 | 0 |
| d5_surface | 1 | bp_lsd | 0/5 | 0.000 | 0/5 | 0.000 | 0 |
| d5_surface | 2 | bp_lsd | 0/5 | 0.000 | 0/5 | 0.000 | 0 |
| d5_surface | 3 | bp_lsd | 1/5 | 0.200 | 1/5 | 0.200 | 0 |
| d5_surface | 4 | bp_lsd | 4/5 | 0.800 | 4/5 | 0.800 | 0 |
| d5_surface | 5 | bp_lsd | 4/5 | 0.800 | 4/5 | 0.800 | 0 |
| d5_surface | 1 | mwpm | 0/5 | 0.000 | - | - | - |
| d5_surface | 2 | mwpm | 0/5 | 0.000 | - | - | - |
| d5_surface | 3 | mwpm | 1/5 | 0.200 | - | - | - |
| d5_surface | 4 | mwpm | 4/5 | 0.800 | - | - | - |
| d5_surface | 5 | mwpm | 4/5 | 0.800 | - | - | - |

## Pipeline agreement (repo vs common, per shot)

| code | decoder | disagreements | n |
| --- | --- | --- | --- |
| d5_surface | bp | 0 | 25 |
| d5_surface | bp_osd | 0 | 25 |
| d5_surface | bp_lsd | 0 | 25 |

## Observation

- Dataset: `create_dataset_from_uniform_shots` puts w X flips AND w Z flips at independent random positions, so a weight-w shot touches up to 2w of the n_data qubits; the nominal weight label comes from the weight-major row layout, not from the shot.
- d5_surface (n_data=25, p_prior=3.84e-03): repo failure fraction by weight 1..5 - bp 0.800/1.000/1.000/1.000/1.000, bp_osd 0.000/0.000/0.200/0.800/0.800, bp_lsd 0.000/0.000/0.200/0.800/0.800, mwpm 0.000/0.000/0.200/0.800/0.800.
- Repo vs common per-shot disagreements: d5_surface bp 0/25; d5_surface bp_osd 0/25; d5_surface bp_lsd 0/25.
