# Sweep report: cartpole_sweep_s0-2

Generated from each run's `config.toml` and `training_log.csv`.
Regenerate with

```bash
python manage.py report \
  --sweep cartpole_sweep_s0-2
```

Do not edit by hand: numbers here and in `summary.csv` must not drift.

## Setup

| Setting | Value |
| :--- | :--- |
| runs | 12 (4 configs, seeds 0, 1, 2) |
| complete | 12 ok, 0 incomplete |
| iterations x episodes | 300 x 5 |
| torch threads | 2 |
| code | `2d71f9611d26` (clean) |
| python / torch / numpy | 3.12.13 / 2.6.0+cpu / 1.26.4 |
| sum of run wall | 11 min |

## Per arm

| arm | seeds | last-100 | solved at | solved | wall |
| :--- | ---: | :--- | :--- | ---: | ---: |
| `01_ppo_mlp` | 3 | 98 +/- 5 | — | 0/3 | 1 +/- 0 min |
| `02_ppo_resmlp` | 3 | 72 +/- 6 | — | 0/3 | 1 +/- 0 min |
| `03_ppo_attn` | 3 | 132 +/- 21 | — | 0/3 | 2 +/- 0 min |
| `04_reinforce` | 3 | 88 +/- 1 | — | 0/3 | 0 +/- 0 min |

`solved at` is the first iteration whose trailing-100 mean reaches 200;
runs that never get there are excluded from that column and counted in
`solved` as unsolved. Values are mean +/- sd across seeds.

## Per run

| arm | seed | last-50 | last-100 | solved at | best | wall | git |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `01_ppo_mlp` | 0 | 115.6 | 101.3 | never | 185.6 | 0.6 min | `2d71f9611d26` |
| `01_ppo_mlp` | 1 | 120.9 | 100.9 | never | 231.8 | 0.6 min | `2d71f9611d26` |
| `01_ppo_mlp` | 2 | 108.0 | 92.2 | never | 192.2 | 0.6 min | `2d71f9611d26` |
| `02_ppo_resmlp` | 0 | 83.4 | 74.2 | never | 146.4 | 0.8 min | `2d71f9611d26` |
| `02_ppo_resmlp` | 1 | 71.3 | 65.1 | never | 123.0 | 0.8 min | `2d71f9611d26` |
| `02_ppo_resmlp` | 2 | 87.0 | 75.9 | never | 166.4 | 0.9 min | `2d71f9611d26` |
| `03_ppo_attn` | 0 | 181.8 | 156.0 | never | 305.2 | 1.8 min | `2d71f9611d26` |
| `03_ppo_attn` | 1 | 130.3 | 114.7 | never | 229.0 | 1.5 min | `2d71f9611d26` |
| `03_ppo_attn` | 2 | 145.0 | 126.5 | never | 248.2 | 1.5 min | `2d71f9611d26` |
| `04_reinforce` | 0 | 106.5 | 87.1 | never | 204.6 | 0.5 min | `2d71f9611d26` |
| `04_reinforce` | 1 | 108.9 | 88.5 | never | 177.2 | 0.5 min | `2d71f9611d26` |
| `04_reinforce` | 2 | 101.1 | 88.6 | never | 186.2 | 0.5 min | `2d71f9611d26` |

## Incomplete

None.
