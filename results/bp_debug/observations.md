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

## O2. Every plain-BP failure is non-convergence; none is a wrong-but-valid correction or an evaluation bug

Step 5 (`step5_convergence_audit.py`, card BD6): iid depolarizing shots at $p=0.003$, 5000 shots per code, repo BP
(`max_iter=30`, repo prior). Each failing shot is put in one class: (a) BP did not converge, (b) converged to a residual that
is syndrome-free but logically wrong, (c) converged, independent check says fine, but the repo's evaluation says failure
(an evaluation bug), (d) other.

| code | n | shots | failing | frac | (a) not converged | (b) converged, logical | (c) repo eval mismatch | (d) other |
|---|---|---|---|---|---|---|---|---|
| d7_surface | 49 | 5000 | 242 | 0.048 | 242 (100%) | 0 | 0 | 0 |
| 288_8_12_tile | 288 | 5000 | 388 | 0.078 | 388 (100%) | 0 | 0 | 0 |
| 144_12_12_ldpc | 144 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 |

Cross-checks: the repo evaluation and the independent logical check agree on every shot (repo-only = check-only = 0 failures), and
re-decoding along the repo path gives the same result (0 disagreements). There are also 0 shots where the `converge` flag is
False but the residual syndrome is zero.

Reading it:

- **H4 (evaluation logic) is ruled out** as a cause of the surface/tile gap: the two ways of counting agree exactly.
- **The failures are BP not settling at all** within 30 iterations, not BP settling on a wrong answer. A logical error would
  need BP to find a syndrome-consistent correction first, and it never does.
- The same arithmetic as F1 still holds at this $p$:

$$
\frac{\text{fail fraction}}{1-(1-p)^n}:\quad
\text{surface } \frac{0.048}{1-0.997^{49}=0.137} = 0.35,\qquad
\text{tile } \frac{0.078}{1-0.997^{288}=0.58} = 0.13 .
$$

  The surface value matches $\bar f_1=0.33$ from O1 (so single errors dominate there). The tile ratio is above its $\bar f_1=0.09$
  because at 288 qubits a typical shot has about $np\approx0.86$ errors, so multi-error shots contribute too.

| code | $np$ at $p=0.003$ | fail frac / P(>=1 error) | O1 $\bar f_1$ |
|---|---|---|---|
| d7_surface | 0.15 | 0.35 | 0.327 |
| 288_8_12_tile | 0.86 | 0.13 | 0.093 |

Figure: `o2_failure_class.png`.

What this suggests, not proves: non-convergence on a boundary single error looks like the oscillation one expects when two
boundary qubits are indistinguishable to BP (same single check). The messages stay symmetric, the hard decision flips between
the two qubits, and the syndrome is never satisfied. This also explains why BP+OSD is fine: OSD does not need BP to converge,
it only needs the soft information to rank columns. Open: confirm by inspecting BP's marginals for one failing boundary qubit
(a small follow-up, not yet a card), and check what damping or min-sum scaling does (step 4).
