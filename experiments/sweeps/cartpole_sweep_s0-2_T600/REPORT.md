# Sweep report: cartpole_sweep_s0-2_T600

Generated from each run's `config.toml` and `training_log.csv`.
Regenerate with

```bash
python manage.py report \
  --sweep cartpole_sweep_s0-2_T600
```

Do not edit by hand: numbers here and in `summary.csv` must not drift.

## Setup

| Setting | Value |
| :--- | :--- |
| runs | 12 (4 configs, seeds 0, 1, 2) |
| complete | 12 ok, 0 incomplete |
| iterations x episodes | 600 x 5 |
| torch threads | 2 |
| code | `1537d418f59b` (clean) |
| python / torch / numpy | 3.12.13 / 2.6.0+cpu / 1.26.4 |
| sum of run wall | 55 min |

## Per arm

| arm | seeds | last-100 | solved at | solved | wall |
| :--- | ---: | :--- | :--- | ---: | ---: |
| `01_ppo_mlp` | 3 | 261 +/- 33 | 486 +/- 47 | 3/3 | 3 +/- 0 min |
| `02_ppo_resmlp` | 3 | 236 +/- 58 | 522 +/- 34 | 2/3 | 4 +/- 1 min |
| `03_ppo_attn` | 3 | 376 +/- 38 | 380 +/- 43 | 3/3 | 9 +/- 1 min |
| `04_reinforce` | 3 | 321 +/- 7 | 427 +/- 6 | 3/3 | 3 +/- 0 min |

`solved at` is the first iteration whose trailing-100 mean reaches 200;
runs that never get there are excluded from that column and counted in
`solved` as unsolved. Values are mean +/- sd across seeds.

## Per run

| arm | seed | last-50 | last-100 | solved at | best | wall | git |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `01_ppo_mlp` | 0 | 240.1 | 241.6 | 507 | 363.4 | 2.5 min | `1537d418f59b` |
| `01_ppo_mlp` | 1 | 323.5 | 300.1 | 432 | 500.0 | 3.0 min | `1537d418f59b` |
| `01_ppo_mlp` | 2 | 259.6 | 242.6 | 519 | 409.8 | 2.4 min | `1537d418f59b` |
| `02_ppo_resmlp` | 0 | 309.4 | 291.4 | 498 | 484.8 | 4.2 min | `1537d418f59b` |
| `02_ppo_resmlp` | 1 | 188.1 | 176.6 | never | 349.4 | 3.3 min | `1537d418f59b` |
| `02_ppo_resmlp` | 2 | 266.2 | 241.1 | 546 | 429.0 | 4.2 min | `1537d418f59b` |
| `03_ppo_attn` | 0 | 416.5 | 413.6 | 336 | 500.0 | 10.2 min | `1537d418f59b` |
| `03_ppo_attn` | 1 | 349.5 | 338.1 | 422 | 500.0 | 7.7 min | `1537d418f59b` |
| `03_ppo_attn` | 2 | 388.1 | 375.9 | 381 | 500.0 | 8.7 min | `1537d418f59b` |
| `04_reinforce` | 0 | 340.6 | 324.7 | 427 | 464.0 | 3.1 min | `1537d418f59b` |
| `04_reinforce` | 1 | 346.6 | 326.4 | 433 | 489.4 | 3.2 min | `1537d418f59b` |
| `04_reinforce` | 2 | 336.3 | 313.1 | 421 | 474.2 | 2.8 min | `1537d418f59b` |

## Incomplete

None.
