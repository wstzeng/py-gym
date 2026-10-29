# Critic ablation results — LunarLander-v3

Sweep `critic_ablation_s0-2`: 8 arms x 3 seeds x 600 iterations, 4 parallel jobs
with 4 threads pinned per job (`taskset` 0-3 / 4-7 / 8-11 / 12-15), 11.9 h wall.

**Numbers live in `REPORT.md`**, which is generated from `summary.csv`; nothing in
this file is typed from memory. Arm definitions (parameters, MACs, tier design)
are in `config/ablation/lunarlander/README.md`. Curves: `comparison.png`.

This run predates provenance recording, so `REPORT.md` reports `code: not
recorded`. Its code was identical to `main` at `d8568cc`, with an unmodified
working tree.

## What the matrix says

Four contrasts, ordered by how clean they are.

1. **`02_mlp_mid` vs `07_resmlp_mid` — normalisation + residual wins.**
   Same parameters (16,897 vs 17,089) *and* same MACs (16,640 vs 16,448); the
   only difference is LayerNorm + residual + GELU. That moves the arm from 0/3
   solved to 2/3, and roughly doubles the trailing-100 reward. Nothing else in
   the matrix is this well controlled.
2. **`06_attn_mid` vs `04_conv_mid` — attention adds nothing measurable.**
   Matched parameters and matched MACs; the two are indistinguishable on both
   iterations-to-solve and final reward, and their seed spread overlaps. Whatever
   the transformer critic was doing, local convolution does it too. Conv also
   costs noticeably less wall time per run at the same nominal FLOPs, because
   attention is a pile of small ops that does not saturate a core.
3. **Capacity is not the explanation, and on its own it hurts.**
   Pure MLPs get *worse* as they grow: 2.1k solves on 2 of 3 seeds, 16.9k solves
   on none, and the 67.6k anchor — the parameter count of the original
   `TransformerCritic` — is the worst arm in the matrix on every seed.
4. **`07_resmlp_mid` vs `06_attn_mid` — still confounded.**
   Parameters match but compute does not: ~16k vs ~141k MACs, 8.6x. The
   token-mixing arm is faster to solve, but this contrast cannot say whether that
   is per-token mixing, the token split itself, or simply 8x the arithmetic.

## Conclusion

The original 3-way comparison (transformer critic solved by ~284 iterations,
plain MLP by ~640, see `experiments/lunarlander_compare/`) does not demonstrate
that self-attention is the reason. What it demonstrates is that the shipped 2.1k
un-normalised MLP critic was both small and under-computed, and that anything
with a normalised residual stream over tokens closes the gap — the cheapest such
thing being a 1D convolution.

## Practical takeaway

If a ~17k critic is wanted, use `04_conv_mid`: same reward as the transformer,
less wall time, and a far smaller implementation. If the budget is ~2.1k
parameters, `03_conv_small` also beats the plain MLP.

## Open question

The remaining confound is compute, not parameters. Two ways to close it, neither
run yet:

- **Sweep `n_tokens` at fixed parameters.** `TransformerCritic` keeps its
  parameter count as `n_tokens` changes; only MACs scale with it
  (`MACs ~ 8 * n_tokens * d_model^2`). `n_tokens = 1` removes token mixing
  entirely, so it doubles as a dense control at matched capacity *and* cost.
  Cheapest informative option: `n_tokens = 1` and `n_tokens = 4` (`06` is `8`).
- **Compute-matched dense control.** `ResidualMLPCritic(hidden=520)` reaches
  ~133k parameters and ~133k MACs, matching `attn_mid` and `conv_mid` in FLOPs
  but at 8x their parameter count.

## Note on cost estimates

MACs are a poor predictor of wall time at this scale. `07_resmlp_mid` and
`02_mlp_mid` have nearly identical MACs but differ ~2.6x in wall time per run,
because LayerNorm + GELU are elementwise work that FLOP counts ignore. Attention
shows the mirror image: `06_attn_mid` costs ~1.5x `04_conv_mid` in wall time at
similar MACs, because transformer blocks decompose into many small kernels.
Estimate from measured wall time, not from MAC count.
