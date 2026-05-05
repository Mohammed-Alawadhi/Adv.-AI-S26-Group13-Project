# Option-2 Evidence Dashboard

# Evaluation Results Summary

Means across 3 seeds, 30 episodes/seed. Error bars = 95% CI (Student-t).


## highway-v0

| Algo | Reward (mean ± 95% CI) | Success Rate | Collision Rate | Episode Length |
|---|---|---|---|---|
| SAC | 136.73 ± 55.08 | 71.1% | 22.2% | 171.2 |
| TDMPC2 | 36.13 ± 24.04 | 1.1% | 92.2% | 37.4 |

## merge-v0

| Algo | Reward (mean ± 95% CI) | Success Rate | Collision Rate | Episode Length |
|---|---|---|---|---|
| SAC | 31.29 ± 25.59 | 0.0% | 85.6% | 34.8 |
| TDMPC2 | 28.94 ± 9.18 | 0.0% | 98.9% | 29.8 |

## roundabout-v0

| Algo | Reward (mean ± 95% CI) | Success Rate | Collision Rate | Episode Length |
|---|---|---|---|---|
| SAC | 25.64 ± 4.24 | 43.3% | 56.7% | 99.3 |
| TDMPC2 | 29.41 ± 3.21 | 53.3% | 46.7% | 120.0 |

## Transfer Gap (relative reward retention from highway-v0)

| Algo | merge-v0 retention | roundabout-v0 retention |
|---|---|---|
| SAC | 22.7% | 19.1% |
| TDMPC2 | 85.3% | 85.4% |

## Statistical Test (Roundabout zero-shot)

- SAC success rate per seed:     [0.4, 0.4, 0.5]
- TD-MPC2 success rate per seed: [0.567, 0.567, 0.467]
- Paired t-test: t=1.500, p=0.2724
- Cohen's d: 0.87
- Mann-Whitney U: U=8.0, p=0.0888

## Dynamics Probe (V1=random, V2=trained-policy; latent MSE @ h=5; lower better)

| Env | V1 MSE @ h=5 | V2 MSE @ h=5 | Random baseline @ h=5 |
|---|---|---|---|
| highway-v0 | 0.0025 | 0.0021 | 0.0103 |
| merge-v0 | 0.0204 | 0.0140 | 0.0007 |
| roundabout-v0 | 0.0283 | 0.0223 | 0.0056 |