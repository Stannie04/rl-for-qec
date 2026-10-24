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

## O3. MWPM is clean, the pipeline is fine, and BP's weight-1 failure rate on uniform shots follows from O1

Step 6 (`step6_reference.py`, card BD7): pymatching MWPM on the same check matrices and shots, with an independent logical-operator
check, on d7_surface and d5_surface. 5000 shots per setting, seed 0.

iid depolarizing:

| code | p | MWPM fails | BP fails | BP frac | BP non-converged |
|---|---|---|---|---|---|
| d7_surface | 0.001 | 0 | 81 | 0.0162 | 81 |
| d7_surface | 0.003 | 0 | 242 | 0.0484 | 242 |
| d7_surface | 0.005 | 0 | 378 | 0.0756 | 378 |
| d5_surface | 0.001 | 0 | 54 | 0.0108 | 54 |
| d5_surface | 0.003 | 0 | 182 | 0.0364 | 182 |
| d5_surface | 0.005 | 0 | 266 | 0.0532 | 266 |

Uniform shots (weight $w$ = $w$ X flips and $w$ Z flips at independent positions):

| code | w | MWPM frac | BP frac |
|---|---|---|---|
| d7_surface | 1 | 0 | 0.437 |
| d7_surface | 2 | 0 | 0.745 |
| d7_surface | 3 | 0 | 0.907 |
| d5_surface | 1 | 0 | 0.542 |
| d5_surface | 2 | 0 | 0.870 |
| d5_surface | 3 | 0.225 | 0.959 |

Figure: `o3_mwpm_vs_bp.png`.

Reading it:

- **The pipeline and the codes are fine.** MWPM has zero failures at every iid point, as it should for $d=7$ and $p\le0.005$.
  The repo's `has_logical_error` and the independent check disagree on 0 of 60000 MWPM predictions. Together with O2 this rules out
  the data, the matrices, the syndrome path and the evaluation. The problem is plain BP.
- **BP failures at iid p are the same as in step 5**: 242 at $p=0.003$ on d7, so runs are reproducible, and again every BP failure
  is non-convergence (the two columns are equal).
- **d5 at w=3 is not a BP bug.** Three X flips exceed what a $d=5$ code corrects ($\lfloor (5-1)/2\rfloor=2$), so MWPM failing 22% there is expected.
  This is the same point as F2: uniform shots at $w\ge 3$ are too dense on small codes to compare decoders fairly.
- **O1 predicts the uniform $w=1$ number.** At $w=1$ there is one X flip and one Z flip, decoded as two independent planes,
  each failing with probability $f$ (the O1 single-Pauli fraction, 0.245 for d7 X or Z):

$$
P(\text{shot fails}) \approx 1-(1-f)^2 = 1-0.755^2 = 0.43
\quad\text{vs measured } 0.437 .
$$

  This shows the same boundary-qubit mechanism drives the heatmap, not just the LER curve.

| code | $f$ (O1) | predicted $1-(1-f)^2$ | measured BP, w=1 |
|---|---|---|---|
| d7_surface | 0.245 | 0.43 | 0.437 |

Caveat: for d5 I have no O1 number yet (step 2 ran on the repo's three codes), so no prediction is made there.

## O4. No BP setting fixes it: 50x more iterations, min-sum, serial schedule and scaling all leave the failure rate where it was

Step 4 (`step4_config_sweep.py`, card BD5): iid depolarizing at $p=0.003$, 5000 shots per code, prior matched to the
true rate ($2p/3=0.002$), one setting changed at a time around the default (`max_iter=30`, product-sum, parallel, scaling 1.0).
All options were checked as valid in `ldpc` 2.4.1. Failure fraction (independent check) equals the non-converged fraction in every row.

| setting | d7_surface | 288_8_12_tile | 144_12_12_ldpc |
|---|---|---|---|
| max_iter = 10 | 0.0484 | 0.0776 | 0 |
| max_iter = 30 (default) | 0.0484 | 0.0776 | 0 |
| max_iter = 100 | 0.0484 | 0.0776 | 0 |
| max_iter = 500 | 0.0484 | 0.0776 | 0 |
| product-sum (default) | 0.0484 | 0.0776 | 0 |
| min-sum, scaling 1.0 | 0.0484 | 0.0774 | 0 |
| min-sum, scaling 0.9 | 0.0484 | 0.0776 | 0 |
| min-sum, scaling 0.75 | 0.0484 | 0.0780 | 0 |
| min-sum, scaling 0.625 | 0.0484 | 0.0784 | 0 |
| min-sum, scaling 0.5 | 0.0522 | 0.1612 | 0 |
| schedule parallel (default) | 0.0484 | 0.0776 | 0 |
| schedule serial | 0.0482 | 0.0772 | 0 |

Figure: `o4_config_sweep.png`.

Reading it:

- **H3 (BP configuration) is ruled out as a fix, and as the cause.** The number of iterations does not matter at all: going from 10
  to 500 gives the identical failure fraction to four digits. If BP were converging slowly, 500 iterations would rescue some shots. It
  rescues none. A shot that has not converged by iteration 10 never converges, so these shots are in a stationary or oscillating
  state from the start, not on a slow path.
- Min-sum and the serial schedule change failure by at most 0.0004, which is within sampling noise for 5000 shots, where

$$
\sigma \approx \sqrt{\frac{\hat q(1-\hat q)}{N}} = \sqrt{\frac{0.0484\cdot 0.9516}{5000}} \approx 0.003 .
$$

- A strong scaling factor (0.5) makes tile much worse (0.078 to 0.161). Under-weighted messages never reach a consistent state.
- The 144_12_12_ldpc code is at zero for every setting, so the gross code is not a sensitive probe for any of this.

Together with O2 (failures are non-convergence) and O3 (MWPM is fine), this says: the decoder fails structurally on boundary
qubits, and no knob in `BpDecoder` that was swept changes that. Not swept: damping or memory-BP (not exposed in this `ldpc` version
as far as I could check), and a matched-prior sweep (step 3, still running). A decoder that does not need BP to converge,
such as BP+OSD or BP+LSD, is the natural fix, and it is what the original three-codes results already show works.

## O5. The BP prior is not the problem (H1 ruled out)

Step 3 (`step3_prior_sweep.py`, card BD4). Same shots, seven priors per shot set: the matched value $2p/3$, $2p$, $p$, the repo value
($2\cdot\text{curriculum}/3$), and fixed 0.01, 0.05, 0.1. Part A: uniform shots, weights 1-3, 5000 shots per cell.
Part B: iid depolarizing, $p=0.001..0.005$, 20000 shots per cell.

Repo prior per code, against the rate it should have at $p=0.001$:

| code | n | repo prior | matched $2p/3$ at p=0.001 | ratio |
|---|---|---|---|---|
| d7_surface | 49 | 0.00196 | 0.00067 | 2.9x too high |
| 288_8_12_tile | 288 | 0.00033 | 0.00067 | 2x too low |
| 144_12_12_ldpc | 144 | 0.00067 | 0.00067 | 1x |

iid shots (BP failure fraction; matched prior vs the extreme prior 0.1, which is up to $0.1/0.00067\approx150$ times too high):

| code | p | matched 2p/3 | repo | 0.1 |
|---|---|---|---|---|
| d7_surface | 0.001 | 0.0159 | 0.0159 | 0.0159 |
| d7_surface | 0.003 | 0.0481 | 0.0481 | 0.0481 |
| d7_surface | 0.005 | 0.0750 | 0.0750 | 0.0750 |
| 288_8_12_tile | 0.001 | 0.0283 | 0.0283 | 0.0283 |
| 288_8_12_tile | 0.003 | 0.0762 | 0.0762 | 0.0767 |
| 288_8_12_tile | 0.005 | 0.1281 | 0.1282 | 0.1287 |

Uniform shots, weight 1 (one X and one Z flip): d7_surface 0.4368 for **every** prior; tile 0.1366 for every prior. At weights 2 and 3 the
priors differ in the third decimal at most (d7 w=3: 0.8988 matched, 0.9298 for $2p$), noise level for 5000 shots.

Figure: `o5_prior_sweep.png`.

Reading it:

- **H1 is ruled out.** With the prior matched exactly to the evaluated $p$, the surface and tile failure is the same as with
  the repo prior or with a prior off by 150x. This was suggested already by O1, where the weight-1 failing sets did not move, and
  it now holds on the real iid distribution too.
- The reason is structural. At weight 1 the BP input is one fired check per plane. The prior only rescales all prior LLRs by the same
  constant $L=\ln\frac{1-q}{q}$, so it cannot break a tie between two qubits whose columns in $H$ are identical. This is the same
  degeneracy story as in O2 and O4, still a hypothesis.
- The worry from F3 (the $144/n$ rescaling of the curriculum rate) is real in the sense that the prior is wrong, but it has no effect on the result.
  I would not touch it for this bug. It could matter if the BP agent is later used as a comparison against a learned agent
  evaluated across $p$, but that is a cleanliness question, not the cause of the 80% error.
- 144_12_12_ldpc stays at 0.0000-0.0001 for every prior in iid, so it cannot tell priors apart.

Status after O1-O5: boundary-qubit single errors fail, BP never converges on them, no prior, no iteration count, no min-sum/schedule
setting fixes it, MWPM is clean. The remaining open items are step 1 (rerun of the repo evaluation on current HEAD, with BP+OSD for
reference) and step 7 (stratified check that the weight-1 bins add up to the reported LER).

## O6. The repo evaluation reproduces the bad BP numbers; BP+OSD, BP+LSD and MWPM are near-perfect on the same shots

Step 1 (`step1_rerun.py`, card BD2): `create_dataset_from_uniform_shots`, 5000 shots per nominal weight $w=1..5$ ($w$ X flips and $w$ Z flips
at independent positions), seed 0. Each shot is decoded two ways: the repo path (`get_agent_and_inference` + `classical_inference`) and the
debug helper (`common.decode_bp`). Failure fraction, repo path:

| code | decoder | w=1 | w=2 | w=3 | w=4 | w=5 |
|---|---|---|---|---|---|---|
| d7_surface | BP | 0.437 | 0.745 | 0.907 | 0.974 | 0.991 |
| d7_surface | BP+OSD | 0 | 0 | 0 | 0.047 | 0.154 |
| d7_surface | BP+LSD | 0 | 0 | 0 | 0.047 | 0.154 |
| d7_surface | MWPM | 0 | 0 | 0 | 0.047 | 0.153 |
| 288_8_12_tile | BP | 0.132 | 0.248 | 0.356 | 0.448 | 0.529 |
| 288_8_12_tile | BP+OSD | 0 | 0 | 0.0004 | 0.0004 | 0.002 |
| 288_8_12_tile | BP+LSD | 0 | 0 | 0.0002 | 0.0004 | 0.002 |
| 144_12_12_ldpc | BP | 0 | 0 | 0.006 | 0.017 | 0.032 |
| 144_12_12_ldpc | BP+OSD | 0 | 0 | 0.005 | 0.016 | 0.027 |
| 144_12_12_ldpc | BP+LSD | 0 | 0 | 0.005 | 0.014 | 0.024 |

Reading it:

- **Reproduced on the current code.** The repo path gives the BP numbers of the three-codes heatmaps (surface 404924/500000 in the
  original run at mixed weights; here 0.44 to 0.99 per weight). The plain BP failure is not an artefact of an old plot (H4, outdated results).
- **The two paths agree for BP.** repo vs `common.decode_bp`: 0 disagreements on all 75000 BP shots (25000 per code). So the debug
  helpers measure the same thing the repo does.
- **Same matrices, same syndromes, three other decoders work.** On surface $w\le3$, BP+OSD, BP+LSD and MWPM fail 0 of 5000. The surface
  $w=4,5$ rows (0.047, 0.154) are the same for all three, and coincide with MWPM, i.e. they are the weights where more than
  $\lfloor(d-1)/2\rfloor=3$ X flips exist and nobody can decode. The corresponding fact in F2: heatmap weight 4-5 on the surface code is simply too dense.
- The gross code is healthy for BP, and OSD/LSD add little there (0.032 vs 0.027 at $w=5$).

Caveats (small, open):

- MWPM returned `error` on the tile and gross codes. Probably these are not matchable (columns of weight >2 in $H$), not investigated.
- BP+OSD repo-vs-helper disagreements: 4/25000 (surface), 4/25000 (tile), 23/25000 (gross); BP+LSD 0, 0, 1. Plain BP: 0.
  Likely OSD tie-breaking differs between the two paths, unexplained, and it does not touch the plain-BP conclusion.

## O7. Stratified by number of flips: at small p, shots with one flip per plane carry most BP failures

Step 7 (`step7_stratified.py`, card BD8): iid depolarizing shots, 200000 per $p\in\{0.001..0.005\}$ per code, repo BP. Bins are
$(n_x,n_z)$ = number of X and Z flips. Failure fraction per bin (selected rows; full table in `step7_stratified.md`):

| code | bin | p=0.001 | p=0.003 | p=0.005 |
|---|---|---|---|---|
| d7_surface | (0,1) | 0.231 | 0.246 | 0.243 |
| d7_surface | (1,0) | 0.254 | 0.250 | 0.249 |
| d7_surface | (1,1) | 0.479 | 0.489 | 0.488 |
| d7_surface | (2,2) | 0.833 | 0.768 | 0.804 |
| 288_8_12_tile | (0,1) | 0.067 | 0.068 | 0.069 |
| 288_8_12_tile | (1,0) | 0.070 | 0.070 | 0.070 |
| 288_8_12_tile | (1,1) | 0.132 | 0.135 | 0.135 |
| 288_8_12_tile | 5+ | 0.302 | 0.338 | 0.353 |
| 144_12_12_ldpc | any bin | 0 | 0 | 0 |

Observations:

- **Failure per bin does not depend on $p$.** The (0,1) bin on the surface code is 0.231, 0.246, 0.243 across a factor 5 in $p$. This is what a purely
  geometric failure (O1) predicts: given a single flip, the chance that it lands on a bad qubit is a property of the code.
- **Bin rates equal the O1 numbers.** The surface (0,1) and (1,0) bins sit at 0.245 and the tile ones at 0.069, which is $12/49$ and $20/288$ from O1.
  The (1,1) bin is almost entirely a Y error on one qubit (a Y is far more likely than an X and a Z on two different qubits), and its rate 0.489 vs 0.490 (O1, Y) and 0.135 vs 0.139 (O1, tile Y) agrees.

$$
\text{surface:}\ \frac{12}{49}=0.245\ \leftrightarrow\ 0.246,\qquad
\frac{24}{49}=0.490\ \leftrightarrow\ 0.489;\qquad
\text{tile:}\ \frac{20}{288}=0.069\ \leftrightarrow\ 0.068,\qquad
\frac{40}{288}=0.139\ \leftrightarrow\ 0.135 .
$$

- **Share of the failures from shots with at most one flip per plane** (bins (0,1), (1,0), (1,1)):

| code | p=0.001 | p=0.003 | p=0.005 |
|---|---|---|---|
| d7_surface | 96.6% | 90% | 82.9% |
| 288_8_12_tile | 78.5% | 49% | 30.6% |

  On the surface code single-flip shots account for almost all failures through $p=0.005$. On the tile code that share falls with $p$ because at 288
  qubits a shot at $p=0.005$ has on average $np\approx1.4$ flips; the multi-flip bins fail at 0.14 to 0.35 and take over.
  Figure: `o7_failure_share.png`.
- **The 'reconstructed LER' check is not an independent test.** The bins partition all shots, so the reconstructed LER equals the direct LER
  to $10^{-20}$ by construction. What the step does show is the per-bin structure above. I should not cite the equality as evidence.
- The all-zero gross-code bins confirm that BP does well on gross even for $(2,2)$ and $5+$ shots at these $p$.

## O8. The weight-1 failing qubits are exactly the qubits whose column in $H$ is duplicated (degeneracy confirmed)

Check (`degeneracy_check.py`, written by the orchestrator, no card): for X errors BP decodes with $H_z$, for Z errors with $H_x$. For each
qubit $j$ I test whether its column $H_{:,j}$ is non-zero and identical to the column of some other qubit. Such a pair gives BP the same
likelihood for either qubit. I compare that set with the step-2 failing sets.

| code | Pauli | qubits with a duplicate column | step-2 failing qubits | in both | dup but not failing | failing but not dup |
|---|---|---|---|---|---|---|
| d7_surface | X | 12 | 12 | 12 | 0 | 0 |
| d7_surface | Z | 12 | 12 | 12 | 0 | 0 |
| 288_8_12_tile | X | 20 | 20 | 20 | 0 | 0 |
| 288_8_12_tile | Z | 20 | 20 | 20 | 0 | 0 |
| 144_12_12_ldpc | X | 0 | 0 | 0 | 0 | 0 |
| 144_12_12_ldpc | Z | 0 | 0 | 0 | 0 | 0 |

The two sets are identical in all six cases, so this is an exact rule, not a correlation:

$$
\text{BP fails on a single error at qubit } j
\iff
\exists\, j'\neq j:\ H_{:,j}=H_{:,j'}\neq 0 .
$$

Mechanism. A weight-2 operator $E_j E_{j'}$ with identical columns has zero syndrome. On a $d=7$ code a zero-syndrome weight-2 operator
cannot be a logical operator, so it is a stabilizer, and either guess ($j$ or $j'$) is a *correct* decode. BP does not know that. With
symmetric priors and symmetric messages the posterior is a coin flip between the two, the hard decision never settles on one qubit, the
syndrome never matches, and `converge=False` (O2). This is also why more iterations (O4) and a different prior (O5) do nothing: the tie is exact,
and neither knob breaks it. OSD or LSD resolves it by a linear-algebra step (O6).

So the BP error is not wrong decoding, it is **BP giving up on a harmless degeneracy**, and the "failure" is counted as an error because the
non-converged residual still has syndrome. The gross code ([[144,12,12]] bivariate bicycle) has no duplicate columns, which is why BP is fine there.

Candidate fixes, all untested (the issue says no fixes here): (1) use BP+OSD or BP+LSD for the surface and tile codes (already working,
O6); (2) break ties in plain BP with a tiny qubit-dependent prior (e.g. jitter the prior by about 1%) or with decimation; (3) report
BP only on codes without duplicate columns. Option 2 is a one-line experiment worth a card.
