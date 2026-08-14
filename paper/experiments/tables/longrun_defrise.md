# defrise: long-run convergence (40 iterations, 256, 25 views / 50 deg)

## Do the two-channel curves meet?

s=1,4,8 share tolerances and objective, so they share a saddle point and must converge to the same RMSE. Single-channel uses a different constraint set (F_hi^2+F_lo^2 != F_s^2) and is NOT expected to join.

| iter | s=1 | s=4 | s=8 | spread | rel. spread |
|---|---|---|---|---|---|
| 40 | 0.25502 | 0.12060 | 0.10383 | 1.51e-01 | 94.60% |

## Iterations to reach a given distance from the shared limit

f* = final iterate of an external reference run (s=4, 40 iterations), so no compared arm defines its own target. Fractions are of each arm's initial distance; all arms start from zero, so the denominators are comparable.

| threshold | two s=1 | two s=4 | two s=8 | single | speedup s=4 vs s=1 |
|---|---|---|---|---|---|
| 50% | 4 | 2 | 4 | 4 | 2.0x |
| 20% | >max | 17 | 13 | >max | n/a |
| 10% | >max | 40 | 27 | >max | n/a |
| 5% | >max | 40 | >max | >max | n/a |
| 2% | >max | 40 | >max | >max | n/a |

## Cost per iteration

| arm | wall-clock (s) | s/iter |
|---|---|---|
| single | 2.3 | 0.0565 |
| two s=1 | 3.3 | 0.0824 |
| two s=4 | 3.4 | 0.0861 |
| two s=8 | 3.3 | 0.0822 |
