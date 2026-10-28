# Critic ablation (LunarLander-v3)

Does the transformer critic win because of **self-attention**, or because it has
**32x the parameters** of the MLP it replaced?

The 3-way comparison behind `config/stable/lunarlander/ppo.toml` compared
`Linear(64->32)` (2,113 parameters) against `TransformerCritic` (67,585
parameters). That conflates three things at once: capacity, normalisation /
residual connections, and the mixing operator. This matrix separates them.

## Design

Everything except `[agent.policy.critic]` is byte-identical across all seven
arms. Do not tune `hyper_params`, `agent.encoder` or `agent.policy` per arm — the
critic is the only variable.

Feature tokens are always `8 x 8` (the encoder emits 64 features). Only the
operator that mixes tokens changes.

| Arm | Mixing | Norm + residual | Params | MACs/sample |
| :--- | :--- | :--- | ---: | ---: |
| `01_mlp_small` | dense | no | 2,113 | 2,080 |
| `03_conv_small` | local (k=3) | yes | 2,269 | 15,714 |
| `05_attn_small` | global (attention) | yes | 2,385 | 2,224 |
| `02_mlp_large` | dense | no | **67,585** | 66,560 |
| `07_resmlp` | dense | yes | 66,625 | 65,600 |
| `04_conv_large` | local (k=3) | yes | 66,769 | **520,104** |
| `06_attn_large` | global (attention) | yes | **67,585** | 66,368 |

`02_mlp_large` and `06_attn_large` have the same parameter count to the unit.

The large tier sits within +/-1.2% of 67,585; the small tier within +/-13% of
2,113. `agent/components/encoder/` tests assert this parity, so a refactor that
breaks it fails `pytest`.

## Reading the result

- `06` vs `02` — the original comparison, now capacity-controlled.
- `06` vs `04` vs `07` — pure mixing-operator comparison (local / global /
  dense), normalisation and residuals held constant.
- `07` vs `06` is the decisive one: if the residual MLP without attention
  matches the transformer, the gain came from **LayerNorm + residual
  connections**, not from attention.
- `01` vs `03` vs `05` — same question at a capacity where a 2k-parameter model
  can barely learn, i.e. is attention an advantage when capacity is scarce.

## Caveats

1. **Conv is not FLOP-matched.** Convolution shares weights across token
   positions, so at equal parameter count it does ~7x the MACs. If `04` wins,
   that may be extra compute rather than better inductive bias. Report both
   columns, always.
2. **`05_attn_small` uses depth 1**, not 2, to stay inside the small parameter
   budget. Attention at 2.4k parameters cannot afford two layers. Capacity parity
   was preferred over depth parity here; the large tier does hold depth constant.
3. **One seed is not a result.** Measured seed noise on this env is large: the
   same `ppo.toml` run twice reached the solved threshold at iteration 530 and
   642. Seeding now exists (`--seed N`, see `utils/seed.py`), so run **>=3 seeds
   per arm** and report `mean +/- sd`. A single-seed arm difference below ~100
   iterations is inside the noise.
4. Training reward is sampled, not deterministic. Comparing arms on it is fair
   but it is not a test-time score.

## Running

```bash
python manage.py train config/ablation/lunarlander/04_conv_large.toml \
  --dry -T 600 -N 20 --record cli --seed 0
```

Always pass `--seed`. Runs sharing a seed see the same per-episode initial
states, so arms differ only in the critic.

600 iterations is deliberate: the baseline reaches the solved threshold at
~640 and the transformer critic at ~290, so 600 keeps the comparison bounded
without hiding a difference.

## Results

Not run yet. Fill in after seeding lands; report `mean +/- sd` across seeds for
iterations-to-solved, final rolling-50, and curve AUC.

| Arm | seeds | iters-to-solved | final rolling-50 | wall time |
| :--- | ---: | :--- | :--- | :--- |
| `01_mlp_small` | 0 | — | — | — |
| `02_mlp_large` | 0 | — | — | — |
| `03_conv_small` | 0 | — | — | — |
| `04_conv_large` | 0 | — | — | — |
| `05_attn_small` | 0 | — | — | — |
| `06_attn_large` | 0 | — | — | — |
| `07_resmlp` | 0 | — | — | — |
