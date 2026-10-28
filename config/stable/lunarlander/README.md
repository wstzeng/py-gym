# LunarLander-v3 stable recipes

## `ppo.toml`

Reference result (2026-10-28, CPU, `OMP_NUM_THREADS=4`, 4 cores):

| Setting | Value |
| :--- | :--- |
| iterations × episodes | 800 × 20 |
| wall time | ~34 min |
| last-100 mean reward | **262.7** (std 14.1) |
| last-50 mean reward | 264.4 |
| best iteration | 284.0 @ 720 |
| solved (trailing-100 ≥ 200) | iteration 530 |

Key changes versus the earlier 500 × 10 run (last-100 mean 24.6):

- `gamma` 0.99 → **0.999** and `gae_lambda` 0.95 → **0.98**: LunarLander episodes
  reach ~1000 steps, so the old horizon was far too short.
- learning rate: `{default: 1e-3, actor: 1e-4}` → **3e-4** for every group.
- `weight_decay` 0.01 → **1e-4**.
- `entropy_weight` 0.01 → **0.001**; sigma still drifted from 0.60 to 0.24, so
  exploration decayed but the policy remained usable.
- episodes per iteration 10 → **20** to reduce reward variance.

Notes:

- The `entropy` column in `training_log.csv` is already weighted by
  `-entropy_weight`; raw entropy is `column / -entropy_weight`.
- Gains flatten after ~600 iterations; 650 is a reasonable early-stop point.
