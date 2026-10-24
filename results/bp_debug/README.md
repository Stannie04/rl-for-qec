# Why is plain BP so bad on the surface and tile codes? (issue #13)

Verdict in one line: the check matrices of the surface and tile codes contain qubits with **identical columns**. BP cannot break the tie
between two such qubits, never converges on a single error at either of them, and the non-converged shot is counted as a failure. The gross code has
no such qubits and is fine. Nothing else in the pipeline is broken.

Full log with every number and plot: `observations.md` (O1-O8). Reproduce with `python -m scripts.bp_debug.<step>`, all steps are in `scripts/bp_debug/`.

| hypothesis | verdict | evidence |
|---|---|---|
| H1 BP prior does not match p | **ruled out** as cause (the prior is wrong, but irrelevant) | O5: matched prior vs prior 150x off, identical failure; O1: failing sets unchanged across priors |
| H2 weight-1 failures | **confirmed**, and explained: boundary qubits with a duplicated column, all three Paulis | O1, O7, O8: failing set == duplicate-column set in all 6 (code, Pauli) cases |
| H3 BP configuration (iterations, min-sum, schedule) | **ruled out** as cause and as fix | O4: 10 vs 500 iterations identical; min-sum and serial change nothing |
| H4 evaluation logic / old results | **ruled out** | O2, O3, O6: repo evaluation and independent logical check agree on every shot; rerun on current code reproduces the numbers |
| H5 independent X/Z decoding | not needed | the failures are fully explained by H2; MWPM, which also decodes X and Z independently, is clean (O3) |
| H6 uniform-weight datasets are dense (F2) | **confirmed as a reading issue** | O6: surface weight 4-5 fails for MWPM too (0.047, 0.15); read the heatmap at $w\le3$ |

Key numbers.

| code | single-error BP failure, mean over X,Y,Z | BP LER / P(>=1 error) | BP+OSD at $w\le3$ | MWPM at $w\le3$ |
|---|---|---|---|---|
| d7_surface | 0.327 | 0.33-0.37 | 0 | 0 |
| 288_8_12_tile | 0.093 | 0.11-0.13 | about 0 | n/a |
| 144_12_12_ldpc | 0 | 5e-6 | 0.005 at w=3 | n/a |

$$\mathrm{LER}_{\mathrm{BP}} \approx \big(1-(1-p)^n\big)\,\bar f_1,\qquad \bar f_1=\tfrac13(f_X+f_Y+f_Z),\qquad f_X = \frac{\#\{j:\ H_{:,j}\ \text{duplicated}\}}{n}.$$

What to do:

1. Do not read "BP" on the surface and tile codes as a decoder-quality baseline. It measures the duplicate-column degeneracy.
2. Untested fixes: BP+OSD/BP+LSD (already fine), a ~1% qubit-dependent prior jitter, decimation. The jitter is the cheapest experiment.
3. Re-cut the surface heatmap at $w\le3$ (O6, F2), and fix the prior to depend on the evaluated $p$ for cleanliness (F3), though it does not matter for this bug.

Open, small: MWPM errors on tile and gross (probably not matchable); BP+OSD disagrees between the repo path and the helper on 4-23 of 25000 shots (OSD tie-break, unexplained, does not touch plain BP).
