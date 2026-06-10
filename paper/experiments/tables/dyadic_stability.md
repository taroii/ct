# Dyadic multi-channel stability diagnostic

Geometry: 256x256, 25 views / 50 deg, 1024 bins. Analytic depth auto_n_channels = 7.

## Certified margin vs dyadic depth

tau*lambda_max(M) for the schedule sigma_i = 2^i, normalized so the uniform schedule (= single-channel) is 1.

| k | tau*lambda_max(M) | certified? |
|---|---|---|
| 2 | 1.161 | no |
| 3 | 1.198 | no |
| 4 | 1.358 | no |
| 5 | 1.799 | no |
| 6 | 2.909 | no |
| 7 | 4.838 | no |

## Per-band spectrum (k = 7)

| band i | energy % | lambda_max(K_i^T K_i) | sigma_i=2^i | sigma_i*lambda_max |
|---|---|---|---|---|
| 0 | 92.57 | 6.438e-01 | 1 | 6.438e-01 |
| 1 | 5.58 | 3.426e-01 | 2 | 6.851e-01 |
| 2 | 1.39 | 2.480e-01 | 4 | 9.919e-01 |
| 3 | 0.35 | 1.792e-01 | 8 | 1.433e+00 |
| 4 | 0.09 | 1.068e-01 | 16 | 1.709e+00 |
| 5 | 0.02 | 1.026e-01 | 32 | 3.282e+00 |
| 6 | 0.01 | 1.404e-01 | 64 | 8.987e+00 |

## Limiting-eigenmode energy share (k = 7)

| band 0 | band 1 | band 2 | band 3 | band 4 | band 5 | band 6 | tv_x | tv_y | l1 |
|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0.001 | 0.010 | 0.071 | 0.825 | 0.000 | 0.000 | 0.093 |
