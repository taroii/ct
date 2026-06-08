# PDHG two-channel stability diagnostic

Geometry: 256x256 image, 25 views / 50 deg arc, 1024 detector bins. Filters: Hann^(1/2) cutoff_hi=4.0, cutoff_lo=8.0.

## Band separation (operator-level, phantom-independent)

| quantity | value |
|---|---|
| lambda_max(X^T F_hi^2 X) | 4.5123 |
| lambda_max(X^T F_lo^2 X) | 0.9230 |
| ratio lo/hi | 0.205 |
| dominant-eigvec overlap \|<v_hi,v_lo>\| | 0.0000 |

## Sharp condition vs sigma_lo_scale

tau*lambda_max(M) = lambda_max(Mtilde(s)) / lambda_max(Mtilde(1)); s=1 is the single-channel boundary (=1).

| sigma_lo_scale | tau*lambda_max(M) | certified (<1)? | lo energy share of limiting mode |
|---|---|---|---|
| 1 | 1.0000 | yes | 0.000 |
| 2 | 1.0000 | yes | 0.000 |
| 4 | 1.0001 | yes | 0.000 |
| 8 | 1.1833 | no | 0.607 |
