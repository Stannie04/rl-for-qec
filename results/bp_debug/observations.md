# BP debug (issue #13): observations log

Each entry: what we saw, the numbers, what it means, what it does not show. Figures are in this folder.

## O1. Plain BP fails single-qubit errors, but only on the code boundary, and the prior does not matter

Step 2 (`step2_weight1.py`, card BD3): one shot per data qubit and per Pauli (X, Y, Z), decoded with the repo's BP
(`max_iter=30`). The shots are deterministic, so there is no sampling noise. I repeated the decode with four different priors
(the repo value, 0.001, 0.01, 0.1).

| code | n | X fails | Y fails | Z fails | mean over X,Y,Z |
|---|---|---|---|---|---|
| d7_surface | 49 | 12/49 = 0.245 | 24/49 = 0.490 | 12/49 = 0.245 | **0.327** |
| 288_8_12_tile | 288 | 20/288 = 0.069 | 40/288 = 0.139 | 20/288 = 0.069 | **0.093** |
| 144_12_12_ldpc | 144 | 0 | 0 | 0 | 0 |

Three things stand out.

1. **The failing qubits are all boundary qubits.** On the surface code every one of the 12 X-failures and 12 Z-failures sits on a
   qubit that touches only one check of the relevant type (the first and last columns, or rows, of the lattice). The same holds
   on the tile code (all 20 listed failing qubits are boundary qubits). Not a single bulk qubit fails.
2. **Y fails exactly where X or Z fails.** Y is decoded as an X half plus a Z half, so it fails if either half fails.
   On the surface code the X-failing and Z-failing sets are disjoint, so $24 = 12 + 12$ exactly.
3. **The prior is irrelevant.** The failing sets are identical for p_prior in {repo, 0.001, 0.01, 0.1}, a factor of 100 in range.

This explains F1. If weight-1 errors dominate at small $p$, then

$$
\mathrm{LER}_{\mathrm{BP}} \;\approx\; P(\ge 1\ \text{error})\cdot \bar f_1,
\qquad
\bar f_1 = \tfrac13\,(f_X + f_Y + f_Z),
\qquad
P(\ge 1\ \text{error}) = 1-(1-p)^n .
$$

| code | $\bar f_1$ (step 2, predicted) | LER / P(>=1 error) (F1, read off plots) |
|---|---|---|
| d7_surface | 0.327 | 0.33 (p=0.001), ~0.37 (p=0.005) |
| 288_8_12_tile | 0.093 | ~0.11 |
| 144_12_12_ldpc | 0 | 5e-6 (healthy) |

Prediction from F1 ("one third of shots with any error fail") now has a mechanism and matches to within the ±20% reading
error of the plots. Figure: `o1_weight1_vs_f1.png`.

What this does and does not show:

- H1 (wrong prior): ruled out as the cause of the weight-1 failures, since failures do not move when the prior changes by 100x.
  It may still matter at higher weights (step 3 will say).
- H2 (weight-1 failures): confirmed. It is not one Pauli type, it is boundary qubits, for all three Pauli types.
- Not yet shown: *why* BP fails on boundary qubits. A natural candidate is degeneracy: two boundary qubits can have the
  same column in $H$ (same single check), so BP cannot tell them apart and may pick the one whose correction differs from the
  true error by a logical operator. That is a hypothesis, not a finding. MWPM (step 6) and the convergence audit (step 5)
  should separate "BP is stuck" from "BP converges to a wrong-but-valid correction".
- Also not shown: whether the step-2 failure criterion is right. Step 6 checks it with an independent logical check.
