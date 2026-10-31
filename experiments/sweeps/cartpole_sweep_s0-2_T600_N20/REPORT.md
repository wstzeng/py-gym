# Sweep report: cartpole_sweep_s0-2_T600_N20

Generated from each run's `config.toml` and `training_log.csv`.
Regenerate with

```bash
python manage.py report \
  --sweep cartpole_sweep_s0-2_T600_N20
```

Do not edit by hand: numbers here and in `summary.csv` must not drift.

## Setup

| Setting | Value |
| :--- | :--- |
| runs | 12 (4 configs, seeds 0, 1, 2) |
| complete | 12 ok, 0 incomplete |
| iterations x episodes | 600 x 20 |
| torch threads | 2 |
| code | `0898e31a0952` (clean) |
| python / torch / numpy | 3.12.13 / 2.6.0+cpu / 1.26.4 |
| sum of run wall | 287 min |

## Per arm

| arm | seeds | last-100 | solved at | solved | wall |
| :--- | ---: | :--- | :--- | ---: | ---: |
| `01_ppo_mlp` | 3 | 324 +/- 4 | 394 +/- 16 | 3/3 | 14 +/- 0 min |
| `02_ppo_resmlp` | 3 | 320 +/- 25 | 448 +/- 26 | 3/3 | 22 +/- 4 min |
| `03_ppo_attn` | 3 | 437 +/- 23 | 310 +/- 42 | 3/3 | 43 +/- 4 min |
| `04_reinforce` | 3 | 455 +/- 7 | 328 +/- 9 | 3/3 | 16 +/- 2 min |

`solved at` is the first iteration whose trailing-100 mean reaches 200;
runs that never get there are excluded from that column and counted in
`solved` as unsolved. Values are mean +/- sd across seeds.

## Per run

| arm | seed | last-50 | last-100 | solved at | best | wall | git |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `01_ppo_mlp` | 0 | 330.4 | 319.5 | 406 | 411.9 | 13.5 min | `0898e31a0952` |
| `01_ppo_mlp` | 1 | 337.5 | 327.1 | 376 | 438.1 | 14.3 min | `0898e31a0952` |
| `01_ppo_mlp` | 2 | 339.0 | 325.9 | 401 | 409.8 | 13.7 min | `0898e31a0952` |
| `02_ppo_resmlp` | 0 | 321.6 | 307.4 | 465 | 406.2 | 18.9 min | `0898e31a0952` |
| `02_ppo_resmlp` | 1 | 322.6 | 304.1 | 462 | 413.1 | 22.4 min | `0898e31a0952` |
| `02_ppo_resmlp` | 2 | 359.0 | 349.3 | 418 | 428.9 | 26.2 min | `0898e31a0952` |
| `03_ppo_attn` | 0 | 456.6 | 449.9 | 296 | 500.0 | 46.9 min | `0898e31a0952` |
| `03_ppo_attn` | 1 | 419.4 | 410.6 | 357 | 465.6 | 39.2 min | `0898e31a0952` |
| `03_ppo_attn` | 2 | 458.7 | 450.3 | 276 | 500.0 | 42.5 min | `0898e31a0952` |
| `04_reinforce` | 0 | 466.6 | 457.0 | 324 | 500.0 | 18.4 min | `0898e31a0952` |
| `04_reinforce` | 1 | 465.7 | 459.9 | 322 | 498.2 | 15.9 min | `0898e31a0952` |
| `04_reinforce` | 2 | 449.3 | 447.2 | 338 | 496.1 | 14.9 min | `0898e31a0952` |

## Incomplete

None.
