# Three-way comparison — LunarLander-v3

The run that motivated `config/ablation/lunarlander/`. Three critics, everything
else identical, 800 iterations x 20 episodes, CPU. Reported in wall-clock order
of the sweep as it was actually run: PPO baseline, PPO + `TransformerCritic`,
plain REINFORCE.

These runs predate seed control, so each is a single unseeded run and the
numbers are not reproducible byte-for-byte.

| Run | Critic | Params (agent) | last-100 | last-50 | best | solved at | wall |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- | ---: |
| `ppo` | MLP 64→32 (2,113) | 6,981 | 255.0 | 257.6 | 285.5 @ 710 | 642 | 54.6 min |
| `transformer` | `TransformerCritic` (67,585) | 72,453 | **278.3** | 279.3 | 295.5 @ 714 | **284** | 131.2 min |
| `reinforce` | MLP 64→32 (2,113) | 4,868 | 11.0 | 9.3 | 34.8 @ 750 | never | 23.8 min |

Configs: `*/config.toml`; curves: `*/training_log.png`, combined in
`comparison.png`. REINFORCE has no baseline and no GAE, which is why it never
leaves single digits — it is included to show what the critic is worth, not as a
tuned arm.

## Why this led to an ablation

The 32x parameter gap between the two PPO critics means the 23-point final
difference and the 358-iteration difference in solve time could come from
capacity, from the transformer's LayerNorm + residual stream, or from attention
itself. The matrix in `config/ablation/lunarlander/` separates those three; see
`experiments/sweeps/critic_ablation_s0-2/RESULTS.md` for the answer, which is
that attention is not the reason.

## Stable recipe reference

`config/stable/lunarlander/ppo.toml` (800 x 20) comes from a separate solo run
with last-100 **262.7** (sd 14.1), last-50 264.4, solved at iteration 530, ~34
minutes. Those artifacts were not preserved; the closest reproducible equivalent
in this directory is the `ppo/` run above (last-100 255.0, solved at 642), which
differs by the expected single-seed noise.
