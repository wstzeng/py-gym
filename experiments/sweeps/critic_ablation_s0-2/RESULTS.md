# Critic ablation results — LunarLander-v3

Sweep `critic_ablation_s0-2`: 8 arms x 3 seeds x 600 iterations, run with

```bash
poetry run python scripts/sweep.py \
  --configs 'config/ablation/lunarlander/*.toml' \
  --seeds 0 1 2 -T 600 -N 20 --jobs 4 --threads 4 --name critic_ablation_s0-2
```

4 parallel jobs, 4 threads pinned per job (`taskset` 0-3 / 4-7 / 8-11 / 12-15).
Wall clock 11.9 h. Arm definitions live in `config/ablation/lunarlander/README.md`.

## Results

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

`last-100` and `solved at` are mean +/- sd over the three seeds; `never` means the
arm never reached a trailing-100 mean of 200 inside 600 iterations. `wall` is the
mean per-run wall time at `OMP_NUM_THREADS=4`. Machine-readable version:
`summary.csv`. Curves: `comparison.png` (redraw with
`poetry run python scripts/plot_sweep.py --sweep experiments/sweeps/critic_ablation_s0-2`).

## What the matrix says

Four contrasts, ordered by how clean they are.

1. **`02_mlp_mid` vs `07_resmlp_mid` — normalisation + residual wins.**
   Same parameters (16,897 vs 17,089) *and* same MACs (16,640 vs 16,448); the
   only difference is LayerNorm + residual + GELU. That moves the arm from
   0/3 solved, last-100 120 +/- 107, to 2/3 solved, 215 +/- 39. Nothing else in
   the matrix is this well controlled.
2. **`06_attn_mid` vs `04_conv_mid` — attention adds nothing measurable.**
   Matched parameters and matched MACs (141k vs 133k), and the two are
   indistinguishable: 269 +/- 17 vs 284 +/- 5 iterations to solve, 240 +/- 6 vs
   260 +/- 4 last-100. Whatever the transformer critic was doing, local
   convolution does it too. Conv also costs 1.5x less wall time per run
   (150 vs 231 min) at the same nominal FLOPs, because attention is a pile of
   small ops that does not saturate a core.
3. **Capacity is not the explanation, and on its own it hurts.**
   Pure MLPs get *worse* as they grow: `01_mlp_small` (2.1k) 212 +/- 36 and 2/3
   solved, `02_mlp_mid` (16.9k) 120 +/- 107 and 0/3, `08_mlp_anchor` (67.6k)
   81 +/- 67 and 0/3. The arm carrying the original `TransformerCritic`'s
   parameter count is the worst arm in the matrix.
4. **`07_resmlp_mid` vs `06_attn_mid` — still confounded.**
   Parameters match (17,089 vs 17,409) but compute does not: 16,448 vs 141,344
   MACs, 8.6x. The token-mixing arm is faster to solve, but this contrast cannot
   say whether that is per-token mixing, the token split itself, or simply 8x the
   arithmetic.

## Conclusion

The original 3-way comparison (transformer critic solved by ~284 iterations,
plain MLP by ~640) does not demonstrate that self-attention is the reason. What
it demonstrates is that the shipped 2.1k un-normalised MLP critic was both
small and under-computed, and that anything with a normalised residual stream
over tokens closes the gap — the cheapest such thing being a 1D convolution.

## Practical takeaway

If a 17k critic is wanted, use `04_conv_mid`: same reward as the transformer,
1.5x less wall time, and a much smaller implementation. If the budget is 2.1k
parameters, `03_conv_small` also beats the plain MLP.

## Open question

The remaining confound is compute, not parameters. Two ways to close it, neither
run yet:

- **Sweep `n_tokens` at fixed parameters** (`TransformerCritic` keeps its
  parameter count as `n_tokens` changes; only MACs scale with it, since
  `MACs ~ 8 * n_tokens * d_model^2`). `n_tokens = 1` removes token mixing
  entirely, so it doubles as a dense control at matched capacity and cost.
  Cheapest informative option: `n_tokens = 1` and `n_tokens = 4` (`06` is `8`).
- **Compute-matched dense control**: `ResidualMLPCritic(hidden=520)` reaches
  ~133k parameters and ~133k MACs, matching `attn_mid` and `conv_mid` in FLOPs
  but at 8x their parameter count.

## Note on cost estimates

MACs are a poor predictor of wall time at this scale. `07_resmlp_mid` and
`02_mlp_mid` have nearly identical MACs (16,448 vs 16,640) but 117 vs 45 minutes
per run, because LayerNorm + GELU are elementwise work that FLOP counts ignore.
Attention shows the mirror image: 141k MACs cost 231 minutes, while conv at
133k MACs costs 150 minutes, because transformer blocks decompose into many
small kernels. Estimate from measured wall time, not from MAC count.
