# LunarLander-v3 stable recipes

## `ppo.toml`

Tuned on CPU (`OMP_NUM_THREADS=4`, 4 cores), 800 iterations x 20 episodes.
Numbers live with the runs, not with the config: see
`experiments/lunarlander_compare/RESULTS.md` for the measured last-100 mean,
solve iteration and wall time, and `experiments/sweeps/critic_ablation_s0-2/`
for how this recipe's critic compares to the alternatives.

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
