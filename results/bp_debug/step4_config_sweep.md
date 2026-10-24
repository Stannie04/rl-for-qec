# Step 4: BP configuration sweep

Dataset: iid depolarizing shots at p=0.003, 5000 shots per code, seed=0; prior fixed at 2p/3=0.0020; one factor at a time around the default (max_iter=30, bp_method=product_sum, ms_scaling_factor=1.0, schedule=parallel). Failure fraction = fraction of shots failing the independent logical check; non-converged fraction = fraction of shots where either the X or the Z BP plane did not converge.

## Validity of ldpc 2.4.1 options

Verified with `help(ldpc.BpDecoder)` and by constructing decoders on ldpc 2.4.1: `bp_method` accepts `product_sum` and `minimum_sum`; `schedule` accepts `parallel` and `serial` (`serial_relative` also exists); `ms_scaling_factor` accepts any float. Every contract value is valid, so no combos were skipped.

## Results

| Code | Factor | Setting | Failure fraction | Non-converged fraction |
|---|---|---|---|---|
| d7_surface | max_iter | max_iter=10 | 0.0484 | 0.0484 |
| d7_surface | max_iter | max_iter=30 | 0.0484 | 0.0484 |
| d7_surface | max_iter | max_iter=100 | 0.0484 | 0.0484 |
| d7_surface | max_iter | max_iter=500 | 0.0484 | 0.0484 |
| d7_surface | bp_method | bp_method=product_sum | 0.0484 | 0.0484 |
| d7_surface | bp_method | bp_method=minimum_sum | 0.0484 | 0.0484 |
| d7_surface | ms_scaling_factor | ms_scaling_factor=0.5 (minimum_sum) | 0.0522 | 0.0522 |
| d7_surface | ms_scaling_factor | ms_scaling_factor=0.625 (minimum_sum) | 0.0484 | 0.0484 |
| d7_surface | ms_scaling_factor | ms_scaling_factor=0.75 (minimum_sum) | 0.0484 | 0.0484 |
| d7_surface | ms_scaling_factor | ms_scaling_factor=0.9 (minimum_sum) | 0.0484 | 0.0484 |
| d7_surface | ms_scaling_factor | ms_scaling_factor=1.0 (minimum_sum) | 0.0484 | 0.0484 |
| d7_surface | schedule | schedule=parallel | 0.0484 | 0.0484 |
| d7_surface | schedule | schedule=serial | 0.0482 | 0.0482 |
| 288_8_12_tile | max_iter | max_iter=10 | 0.0776 | 0.0776 |
| 288_8_12_tile | max_iter | max_iter=30 | 0.0776 | 0.0776 |
| 288_8_12_tile | max_iter | max_iter=100 | 0.0776 | 0.0776 |
| 288_8_12_tile | max_iter | max_iter=500 | 0.0776 | 0.0776 |
| 288_8_12_tile | bp_method | bp_method=product_sum | 0.0776 | 0.0776 |
| 288_8_12_tile | bp_method | bp_method=minimum_sum | 0.0774 | 0.0774 |
| 288_8_12_tile | ms_scaling_factor | ms_scaling_factor=0.5 (minimum_sum) | 0.1612 | 0.1612 |
| 288_8_12_tile | ms_scaling_factor | ms_scaling_factor=0.625 (minimum_sum) | 0.0784 | 0.0784 |
| 288_8_12_tile | ms_scaling_factor | ms_scaling_factor=0.75 (minimum_sum) | 0.0780 | 0.0780 |
| 288_8_12_tile | ms_scaling_factor | ms_scaling_factor=0.9 (minimum_sum) | 0.0776 | 0.0776 |
| 288_8_12_tile | ms_scaling_factor | ms_scaling_factor=1.0 (minimum_sum) | 0.0774 | 0.0774 |
| 288_8_12_tile | schedule | schedule=parallel | 0.0776 | 0.0776 |
| 288_8_12_tile | schedule | schedule=serial | 0.0772 | 0.0772 |
| 144_12_12_ldpc | max_iter | max_iter=10 | 0.0000 | 0.0000 |
| 144_12_12_ldpc | max_iter | max_iter=30 | 0.0000 | 0.0000 |
| 144_12_12_ldpc | max_iter | max_iter=100 | 0.0000 | 0.0000 |
| 144_12_12_ldpc | max_iter | max_iter=500 | 0.0000 | 0.0000 |
| 144_12_12_ldpc | bp_method | bp_method=product_sum | 0.0000 | 0.0000 |
| 144_12_12_ldpc | bp_method | bp_method=minimum_sum | 0.0000 | 0.0000 |
| 144_12_12_ldpc | ms_scaling_factor | ms_scaling_factor=0.5 (minimum_sum) | 0.0000 | 0.0000 |
| 144_12_12_ldpc | ms_scaling_factor | ms_scaling_factor=0.625 (minimum_sum) | 0.0000 | 0.0000 |
| 144_12_12_ldpc | ms_scaling_factor | ms_scaling_factor=0.75 (minimum_sum) | 0.0000 | 0.0000 |
| 144_12_12_ldpc | ms_scaling_factor | ms_scaling_factor=0.9 (minimum_sum) | 0.0000 | 0.0000 |
| 144_12_12_ldpc | ms_scaling_factor | ms_scaling_factor=1.0 (minimum_sum) | 0.0000 | 0.0000 |
| 144_12_12_ldpc | schedule | schedule=parallel | 0.0000 | 0.0000 |
| 144_12_12_ldpc | schedule | schedule=serial | 0.0000 | 0.0000 |

## Observation

- Default (product_sum, parallel, ms_scaling_factor=1.0, max_iter=30): failure fraction d7_surface=0.0484, 288_8_12_tile=0.0776, 144_12_12_ldpc=0.0000.
- max_iter 30->500 failure: d7_surface 0.0484->0.0484, 288_8_12_tile 0.0776->0.0776, 144_12_12_ldpc 0.0000->0.0000.
- bp_method product_sum->minimum_sum failure: d7_surface 0.0484->0.0484, 288_8_12_tile 0.0776->0.0774, 144_12_12_ldpc 0.0000->0.0000.
- schedule parallel->serial failure: d7_surface 0.0484->0.0482, 288_8_12_tile 0.0776->0.0772, 144_12_12_ldpc 0.0000->0.0000.
- ms_scaling_factor (minimum_sum, 0.5..1.0) failure range: d7_surface 0.0484..0.0522, 288_8_12_tile 0.0774..0.1612, 144_12_12_ldpc 0.0000..0.0000.
