# BP debug: findings so far (partial, before any card results)

Status: derived from reading `origin/paper` and from arithmetic on the configs. No decoder has been re-run yet. Reported BP
numbers are read by eye off Stan's plots (`results/results_three_codes/`), so treat them as ±20%.
Reproduce the facts table: `PYTHONPATH=. python` on `results/bp_debug/facts.json` (made from `configs/`).

## F1. BP's LER is a fixed fraction of "shots with at least one error"  (strong evidence, H2)

Per-shot probability that at least one qubit is hit under iid depolarizing noise: `1-(1-p)^n`.

| code | n | p | P(>=1 error) | reported BP LER | ratio |
|---|---|---|---|---|---|
| d7_surface | 49 | 0.001 | 0.048 | 1.6e-2 | **0.33** |
| d7_surface | 49 | 0.005 | 0.218 | ~8e-2 | **~0.37** |
| 288_8_12_tile | 288 | 0.001 | 0.250 | 2.7e-2 | **~0.11** |
| 144_12_12_ldpc | 144 | 0.001 | 0.134 | 7e-7 | 5e-6 (healthy) |

Plot: `finding1_bp_ler_vs_any_error.png`. A d=7 code corrects any 3 errors, so a working decoder would give an LER around 1e-8
at p=0.001 on the surface code. BP's LER is instead exactly one third of the shots that contain any error, at two different p.
That ratio is independent of p, which means almost every low-weight error pattern of a certain type fails, not just rare
high-weight ones. The tile code looks like the same mechanism with a smaller fraction (about 1/9).
Reading: a systematic failure of single-qubit (or very low-weight) errors, e.g. all errors of one Pauli type, or some qubits.
Not yet shown: which Pauli, which qubits. That is step 2 (BD3).

## F2. The uniform "weight" datasets are very dense on the small code  (explains the heatmap, not the LER curves)

`create_dataset_from_uniform_shots` puts w X errors and w Z errors at independent positions. Expected fraction of qubits touched:

| code | n | d | w=1 | w=2 | w=3 | w=4 | w=5 |
|---|---|---|---|---|---|---|---|
| d7_surface | 49 | 7 | 4.0% | 8.0% | 11.9% | 15.7% | 19.4% |
| 288_8_12_tile | 288 | 12 | 0.7% | 1.4% | 2.1% | 2.8% | 3.4% |
| 144_12_12_ldpc | 144 | 12 | 1.4% | 2.8% | 4.1% | 5.5% | 6.8% |

Plot: `finding2_uniform_density.png`. On the surface code, weight 4-5 means 5 X flips against a correctable limit of 3, so any
decoder fails often there. This inflates the surface heatmap numbers and the surface/gross comparison. It does not explain the LER
curves, which use iid noise at p <= 0.005 where weight >= 4 errors are rare (expected X flips per shot at p=0.001: 0.033).
Consequence: the heatmap should be re-read per weight, and a surface-code heatmap restricted to w <= 3 is the fair one.

## F3. BP prior is fixed to the curriculum rate, not the evaluated p  (real, but probably not the main cause)

`BPAgent`: `error_rate = 2*curriculum_error_rate/3`, `curriculum_error_rate = 0.001*144/n_data`.

| code | n_data | prior used (per X-type flip) | matched `2p/3` at p=0.001 | at p=0.005 | mismatch range |
|---|---|---|---|---|---|
| d7_surface | 49 | 0.00196 | 0.00067 | 0.0033 | 2.9x too high ... 1.7x too low |
| 288_8_12_tile | 288 | 0.00033 | 0.00067 | 0.0033 | 2x ... 10x too low |
| 144_12_12_ldpc | 144 | 0.00067 | 0.00067 | 0.0033 | 1x ... 5x too low |

A prior within a factor of 3 of the truth should not make weight-1 errors fail on a d=7 surface code, so H1 alone cannot explain F1.
It can still distort the high-p end for tile and gross. Not yet tested (step 3, BD4).

## F4. Code and decoder inputs look sane  (H matrices not suspected)

| code | H_x shape | H_z shape | k | d |
|---|---|---|---|---|
| d7_surface | 24 x 49 | 24 x 49 | 1 | 7 |
| 288_8_12_tile | 140 x 288 | 140 x 288 | 8 | 12 |
| 144_12_12_ldpc | 72 x 144 | 72 x 144 | 12 | 12 |

BP+OSD and BP+LSD (same matrices, same syndromes) reach about 4% on the surface heatmap, so the matrices and the syndrome
construction in `ClassicAgent.select_action` are consistent. The failure is specific to plain BP (setup or convergence) or to the
BP-only evaluation path. Note that the surface code and tile code have equal numbers of X and Z checks only because of
construction; the code handles unequal numbers correctly.

## F5. Process note

The contract first said `ConfigParser(..., "bp", ...)`; `model_config.yml` only has `sac`, `moe`, `encoder`. Fixed in
`scripts/bp_debug/INTERFACES.md` (agent name `"sac"`). BD1's worktree started from the old contract.
