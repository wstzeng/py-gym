# Critic ablation (LunarLander-v3)

Does the transformer critic win because of **self-attention**, or because it has
**32x the parameters** of the MLP it replaced?

The 3-way comparison behind `config/stable/lunarlander/ppo.toml` compared
`Linear(64->32)` (2,113 parameters) against `TransformerCritic` (67,585
parameters). That conflates three things at once: capacity, normalisation /
residual connections, and the mixing operator. This matrix separates them.

## Design

Everything except `[agent.policy.critic]` is byte-identical across all eight
arms. Do not tune `hyper_params`, `agent.encoder` or `agent.policy` per arm — the
critic is the only variable.

Feature tokens are always `8 x 8` (the encoder emits 64 features). Only the
operator that mixes tokens changes.

| Arm | Tier | Mixing | Norm + residual | Params | MACs/sample |
| :--- | :--- | :--- | :--- | ---: | ---: |
| `01_mlp_small` | small | dense | no | 2,113 | 2,080 |
| `03_conv_small` | small | local (k=3) | yes | 2,269 | 16,722 |
| `05_attn_small` | small | global (attention) | yes | 2,385 | 19,472 |
| `02_mlp_mid` | mid | dense | no | 16,897 | 16,640 |
| `07_resmlp_mid` | mid | dense | yes | 17,089 | 16,448 |
| `04_conv_mid` | mid | local (k=3) | yes | 17,161 | 133,172 |
| `06_attn_mid` | mid | global (attention) | yes | 17,409 | 141,344 |
| `08_mlp_anchor` | anchor | dense | no | **67,585** | 66,560 |

`01_mlp_small` is exactly the critic the shipped stable recipe uses, so the
small tier keeps a link to an already validated run. The mid tier is the matrix
proper: four mixing families at the same capacity. `08_mlp_anchor` carries the
parameter count of the original `TransformerCritic` result as a pure MLP, which
makes the "was it just capacity?" question answerable without paying for a
67k attention arm.

Parity: mid within +/-2% of 17,150, small within +/-13% of 2,113, anchor exact.
`tests/test_critic_ablations.py` asserts this, so a refactor that breaks it
fails `pytest`.

## Reading the result

- `06` vs `02` — capacity-controlled version of the original comparison.
- `06` vs `04` vs `07` — pure mixing-operator comparison (global / local /
  dense), normalisation and residuals held constant.
- `07` vs `06` is the decisive one: if the residual MLP without attention
  matches the transformer, the gain came from **LayerNorm + residual
  connections**, not from attention.
- `08` vs the earlier 67,585-parameter transformer run — does a 4x cheaper MLP
  of the same size reach the same place?
- `01` vs `03` vs `05` — same question at a capacity where a 2k-parameter model
  can barely learn, i.e. is attention an advantage when capacity is scarce.

## Caveats

1. **Capacity is matched, compute is not.** The token-mixing arms (conv,
   attention) apply every weight to all 8 tokens, so they do ~8x the MACs of the
   vector arms (MLP, residual MLP) at the same parameter count. Conv and
   attention *are* roughly FLOP-matched to each other (133k vs 141k at mid), so
   `06` vs `04` is a clean operator comparison, but `06` vs `07` also differs in
   compute by 8x. Report both columns, always.
2. **`05_attn_small` uses depth 1**, not 2, to stay inside the small parameter
   budget. Attention at 2.4k parameters cannot afford two layers. Capacity parity
   was preferred over depth parity here; the mid tier does hold depth constant.
3. **One seed is not a result.** Measured seed noise on this env is large: the
   same `ppo.toml` run twice reached the solved threshold at iteration 530 and
   642. Seeding exists (`--seed N`, see `utils/seed.py`), so run **>=3 seeds per
   arm** and report `mean +/- sd`. A single-seed arm difference below ~100
   iterations is inside the noise. Because the mid tier squeezes the arms closer
   together, expect the effect size to shrink too; if it lands inside the noise,
   extend to 5 seeds before concluding anything.
4. Training reward is sampled, not deterministic. Comparing arms on it is fair
   but it is not a test-time score.

## Running

One arm, one seed:

```bash
poetry run python manage.py train config/ablation/lunarlander/06_attn_mid.toml \
  --dry -T 600 -N 20 --record cli --seed 0
```

The whole matrix (`scripts/sweep.py` keeps directories apart and writes
`experiments/sweeps/<name>/summary.csv`):

```bash
poetry run python scripts/sweep.py \
  --configs 'config/ablation/lunarlander/*.toml' \
  --seeds 0 1 2 -T 600 -N 20 --jobs 4 --threads 4 --name critic_ablation_s0-2
```

Always pass `--seed`. Runs sharing a seed see the same per-episode initial
states, so arms differ only in the critic. Pin `OMP_NUM_THREADS` (the sweep does
it for you) or bit-identical reruns are not guaranteed.

600 iterations is deliberate: the baseline reaches the solved threshold at
~640 and the transformer critic at ~290, so 600 keeps the comparison bounded
without hiding a difference. Budget ~12 hours for 8 arms x 3 seeds at 4 parallel
jobs; the critic-heavy arms cost far more than their MAC count suggests, because
attention and convolution are many small ops rather than a few large matmuls.

## Results

Run `critic_ablation_s0-2`: 8 arms x 3 seeds x 600 iterations, `--jobs 4
--threads 4`, 4 threads pinned per run. 11.9 hours wall. Raw artifacts in
`experiments/sweeps/critic_ablation_s0-2/` (untracked), curves in
`comparison.png` next to this file's run directory.

| Arm | Params | MACs | last-100 | solved at | solved | wall |
| :--- | ---: | ---: | :--- | :--- | ---: | ---: |
| `01_mlp_small` | 2,113 | 2,080 | 212 +/- 36 | 540 +/- 23 | 2/3 | 44 min |
| `03_conv_small` | 2,269 | 16,722 | 238 +/- 16 | 303 +/- 35 | 3/3 | 126 min |
| `05_attn_small` | 2,385 | 19,472 | 233 +/- 17 | 341 +/- 63 | 3/3 | 174 min |
| `02_mlp_mid` | 16,897 | 16,640 | 120 +/- 107 | never | 0/3 | 45 min |
| `07_resmlp_mid` | 17,089 | 16,448 | 215 +/- 39 | 499 +/- 41 | 2/3 | 117 min |
| `04_conv_mid` | 17,161 | 133,172 | 260 +/- 4 | 284 +/- 5 | 3/3 | 150 min |
| `06_attn_mid` | 17,409 | 141,344 | 240 +/- 6 | 269 +/- 17 | 3/3 | 231 min |
| `08_mlp_anchor` | 67,585 | 66,560 | 81 +/- 67 | never | 0/3 | 45 min |

`never` = did not reach a trailing-100 mean of 200 within 600 iterations. Wall
is mean per run at `OMP_NUM_THREADS=4`.

## What the matrix says

Four contrasts, ordered by how clean they are:

1. **`02_mlp_mid` vs `07_resmlp_mid` — normalisation + residuals win.**
   Same parameters (16,897 vs 17,089) *and* same MACs (16,640 vs 16,448); the
   only difference is LayerNorm + residual + GELU. That moves the arm from
   0/3 solved, last-100 120 +/- 107, to 2/3 solved, 215 +/- 39. Nothing else in
   the matrix is this well controlled.
2. **`06_attn_mid` vs `04_conv_mid` — attention adds nothing.**
   Matched parameters and matched MACs (141k vs 133k), and the two are
   indistinguishable: 269 +/- 17 vs 284 +/- 5 iterations to solve, 240 +/- 6 vs
   260 +/- 4 last-100. Whatever the transformer critic was doing, local
   convolution does it too. Conv also costs 1.5x less wall time per run
   (150 vs 231 min) for the same nominal FLOPs, because attention is a pile of
   small ops that do not saturate a core.
3. **Capacity is not the explanation, and on its own it hurts.**
   Pure MLPs get *worse* as they grow: `01_mlp_small` (2.1k) 212 +/- 36 and 2/3
   solved, `02_mlp_mid` (16.9k) 120 +/- 107 and 0/3, `08_mlp_anchor` (67.6k)
   81 +/- 67 and 0/3. The arm that carries the original `TransformerCritic`'s
   parameter count is the worst arm in the matrix.
4. **`07_resmlp_mid` vs `06_attn_mid` — still confounded.**
   Parameters match (17,089 vs 17,409) but compute does not: 16,448 vs 141,344
   MACs, 8.6x. The transformer is faster to solve, but this contrast cannot say
   whether that is attention, per-token mixing, or simply 8x the arithmetic.
   A compute-matched dense control is the one missing cell; see below.

The original 3-way result (transformer critic solved at ~284, plain MLP at
~640) therefore does **not** demonstrate that self-attention is the reason. It
demonstrates that the shipped 2.1k un-normalised MLP critic was under-powered and
under-computed, and that anything with a normalised residual stream over tokens
closes the gap.

### Missing cell

To separate "per-token mixing" from "8x the FLOPs", the dense arm needs to be
compute-matched rather than parameter-matched: `ResidualMLPCritic(hidden=520)`
lands at ~133k parameters and ~133k MACs, i.e. the same FLOPs as `attn_mid` and
`conv_mid`, but 8x the parameters of the other mid arms. That arm is not part of
the tracked matrix because it breaks parameter parity by construction.
