# Sweep report: cartpole_sweep_s0-2_T800

Generated from each run's `config.toml` and `training_log.csv`.
Regenerate with

```bash
python manage.py report \
  --sweep cartpole_sweep_s0-2_T800
```

Do not edit by hand: numbers here and in `summary.csv` must not drift.

## Setup

| Setting | Value |
| :--- | :--- |
| runs | 12 (4 configs, seeds 0, 1, 2) |
| complete | 12 ok, 0 incomplete |
| iterations x episodes | 800 x 5 |
| torch threads | 2 |
| code | `992bf89ccc1c` (clean) |
| python / torch / numpy | 3.12.13 / 2.6.0+cpu / 1.26.4 |
| sum of run wall | 119 min |

## Per arm

| arm | seeds | last-100 | solved at | solved | wall |
| :--- | ---: | :--- | :--- | ---: | ---: |
| `01_ppo_mlp` | 3 | 348 +/- 17 | 486 +/- 47 | 3/3 | 6 +/- 1 min |
| `02_ppo_resmlp` | 3 | 350 +/- 63 | 575 +/- 95 | 3/3 | 9 +/- 2 min |
| `03_ppo_attn` | 3 | 436 +/- 3 | 380 +/- 43 | 3/3 | 16 +/- 1 min |
| `04_reinforce` | 3 | 441 +/- 13 | 427 +/- 6 | 3/3 | 7 +/- 0 min |

`solved at` is the first iteration whose trailing-100 mean reaches 200;
runs that never get there are excluded from that column and counted in
`solved` as unsolved. Values are mean +/- sd across seeds.

## Per run

| arm | seed | last-50 | last-100 | solved at | best | wall | git |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `01_ppo_mlp` | 0 | 350.1 | 346.6 | 507 | 500.0 | 6.3 min | `992bf89ccc1c` |
| `01_ppo_mlp` | 1 | 373.9 | 366.0 | 432 | 500.0 | 7.1 min | `992bf89ccc1c` |
| `01_ppo_mlp` | 2 | 340.4 | 332.1 | 519 | 489.0 | 6.0 min | `992bf89ccc1c` |
| `02_ppo_resmlp` | 0 | 411.3 | 402.5 | 498 | 500.0 | 11.2 min | `992bf89ccc1c` |
| `02_ppo_resmlp` | 1 | 292.9 | 280.1 | 682 | 475.8 | 7.1 min | `992bf89ccc1c` |
| `02_ppo_resmlp` | 2 | 370.7 | 367.4 | 546 | 500.0 | 10.1 min | `992bf89ccc1c` |
| `03_ppo_attn` | 0 | 433.4 | 438.7 | 336 | 500.0 | 17.7 min | `992bf89ccc1c` |
| `03_ppo_attn` | 1 | 442.4 | 436.2 | 422 | 500.0 | 15.4 min | `992bf89ccc1c` |
| `03_ppo_attn` | 2 | 430.5 | 433.5 | 381 | 500.0 | 16.3 min | `992bf89ccc1c` |
| `04_reinforce` | 0 | 429.6 | 433.3 | 427 | 500.0 | 7.5 min | `992bf89ccc1c` |
| `04_reinforce` | 1 | 459.8 | 456.0 | 433 | 500.0 | 7.2 min | `992bf89ccc1c` |
| `04_reinforce` | 2 | 448.8 | 432.3 | 421 | 500.0 | 6.9 min | `992bf89ccc1c` |

## Incomplete

None.
