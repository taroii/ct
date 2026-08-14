# defrise: Poisson-noise statistics (i0=8e+06, 8 realizations, s=4)

Paired: both arms reconstruct the same realization, so the difference removes shared realization-to-realization variation. Solver seed is fixed at 42 across all realizations; only the noise seed varies, so the spread below is measurement noise and not step-size jitter.

eps calibrated to the noise from an independent draw: 0.04227 +/- 0.00063 (noiseless value would be 0.001). Zero-count rays: 0.00e+00 worst case.

Positive difference = two-channel has LOWER RMSE.

| iter | single RMSE | two RMSE | mean diff | 95% CI | median [IQR] | win rate | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| 10 | 0.39058 | 0.15689 | +0.23368 (+59.8%) | [+0.23306, +0.23431] | +0.23398 [+0.23297, +0.23428] | 100% | 0.0078 |
| 50 | 0.21837 | 0.14456 | +0.07381 (+33.8%) | [+0.07327, +0.07434] | +0.07384 [+0.07360, +0.07423] | 100% | 0.0078 |
| 60 | 0.17146 | 0.14268 | +0.02878 (+16.8%) | [+0.02801, +0.02955] | +0.02878 [+0.02833, +0.02946] | 100% | 0.0078 |

A 95% CI that excludes zero means the difference at that iteration is resolved by this many realizations; one that straddles zero means it is not, regardless of the point estimate's sign.
