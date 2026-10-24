# Step 5: convergence / failure-class audit

Dataset: iid depolarizing shots, `p=0.003`, `5000` shots per code, seed `0`. Decoder: repo-default plain BP (`common.decode_bp`, `p_prior=2*curriculum_error_rate/3`, `max_iter=30`). "Failing" = union of the independent check and the repo evaluation; (a) not effectively converged (flag False *and* syndrome non-free), (b) converged + syndrome-free + independent logical failure, (c) converged + syndrome-free + independent success but repo failure, (d) other.

| code | n | shots | failing | frac | (a) not converged | (b) converged, logical | (c) repo eval mismatch | (d) other |
|---|---|---|---|---|---|---|---|---|
| d7_surface | 49 | 5000 | 242 | 0.048 | 242 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| 288_8_12_tile | 288 | 5000 | 388 | 0.078 | 388 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| 144_12_12_ldpc | 144 | 5000 | 0 | 0.000 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |

## Failure-class examples (shot indices)

| code | class | count | example shot indices |
|---|---|---|---|
| d7_surface | (a) not converged | 242 | [11, 21, 23] |
| 288_8_12_tile | (a) not converged | 388 | [1, 17, 18] |

## Diagnostics

| code | independent failures | repo failures | repo-only | check-only | repo path re-decoded | repo path disagreements | non-converged x | non-converged z | flag/syndrome mismatch x | flag/syndrome mismatch z |
|---|---|---|---|---|---|---|---|---|---|---|
| d7_surface | 242 | 242 | 0 | 0 | 242 | 0 | 113 | 133 | 0 | 0 |
| 288_8_12_tile | 388 | 388 | 0 | 0 | 388 | 0 | 193 | 203 | 0 | 0 |
| 144_12_12_ldpc | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Observation

- `d7_surface`: 242/5000 shots fail (0.048); the largest class is (a) not converged with 242 shots (100% of the failing set).
- `288_8_12_tile`: 388/5000 shots fail (0.078); the largest class is (a) not converged with 388 shots (100% of the failing set).
- `144_12_12_ldpc`: no failing shot in 5000 shots.
- `d7_surface`: BP did not converge on 113 x-plane and 133 z-plane decodes out of 5000; 0 x and 0 z of those are flag/syndrome mismatches (flag False while the residual syndrome is free, hence effectively converged); 0 shots fail the repo evaluation only and 0 the independent check only.
- `288_8_12_tile`: BP did not converge on 193 x-plane and 203 z-plane decodes out of 5000; 0 x and 0 z of those are flag/syndrome mismatches (flag False while the residual syndrome is free, hence effectively converged); 0 shots fail the repo evaluation only and 0 the independent check only.
