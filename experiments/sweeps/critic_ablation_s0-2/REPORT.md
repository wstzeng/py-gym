# Sweep report: critic_ablation_s0-2

Generated from `summary.csv` and each run's `run.json` / `config.json`.
Regenerate with

```bash
poetry run python scripts/sweep.py \
  --report-only --name critic_ablation_s0-2
```

Do not edit by hand: numbers here and in `summary.csv` must not drift.

## Setup

| Setting | Value |
| :--- | :--- |
| runs | 24 (8 configs, seeds 0, 1, 2) |
| completed | 24 ok, 0 failed |
| iterations x episodes | 600 x 20 |
| torch threads | 4 |
| code | not recorded (run predates provenance) |
| python / torch / numpy | — / — / — |
| sum of run wall | 2800 min |

## Per arm

| arm | seeds | last-100 | solved at | solved | wall |
| :--- | ---: | :--- | :--- | ---: | ---: |
| `01_mlp_small` | 3 | 212 +/- 36 | 540 +/- 23 | 2/3 | 44 +/- 8 min |
| `02_mlp_mid` | 3 | 120 +/- 107 | — | 0/3 | 45 +/- 12 min |
| `03_conv_small` | 3 | 238 +/- 16 | 303 +/- 35 | 3/3 | 126 +/- 4 min |
| `04_conv_mid` | 3 | 260 +/- 4 | 284 +/- 5 | 3/3 | 150 +/- 15 min |
| `05_attn_small` | 3 | 233 +/- 17 | 341 +/- 63 | 3/3 | 174 +/- 36 min |
| `06_attn_mid` | 3 | 240 +/- 6 | 269 +/- 17 | 3/3 | 231 +/- 20 min |
| `07_resmlp_mid` | 3 | 215 +/- 39 | 499 +/- 41 | 2/3 | 117 +/- 2 min |
| `08_mlp_anchor` | 3 | 81 +/- 67 | — | 0/3 | 45 +/- 15 min |

`solved at` is the first iteration whose trailing-100 mean reaches 200;
runs that never get there are excluded from that column and counted in
`solved` as failures. Values are mean +/- sd across seeds.

## Per run

| arm | seed | rc | last-50 | last-100 | solved at | best | wall | git |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `01_mlp_small` | 0 | 0 | 242.0 | 229.6 | 557 | 272.0 | 53.2 min | `—` |
| `01_mlp_small` | 1 | 0 | 199.6 | 170.4 | never | 251.9 | 38.2 min | `—` |
| `01_mlp_small` | 2 | 0 | 241.6 | 235.2 | 524 | 276.0 | 39.5 min | `—` |
| `02_mlp_mid` | 0 | 0 | 198.4 | 188.5 | never | 242.4 | 48.7 min | `—` |
| `02_mlp_mid` | 1 | 0 | 9.4 | -2.5 | never | 42.6 | 31.9 min | `—` |
| `02_mlp_mid` | 2 | 0 | 198.5 | 175.1 | never | 246.8 | 55.1 min | `—` |
| `03_conv_small` | 0 | 0 | 231.7 | 227.7 | 278 | 260.2 | 121.8 min | `—` |
| `03_conv_small` | 1 | 0 | 236.9 | 230.6 | 343 | 259.7 | 126.8 min | `—` |
| `03_conv_small` | 2 | 0 | 252.1 | 255.9 | 289 | 273.9 | 130.4 min | `—` |
| `04_conv_mid` | 0 | 0 | 259.0 | 255.9 | 279 | 285.1 | 166.6 min | `—` |
| `04_conv_mid` | 1 | 0 | 263.4 | 259.9 | 289 | 282.3 | 136.5 min | `—` |
| `04_conv_mid` | 2 | 0 | 265.5 | 262.9 | 285 | 282.7 | 148.0 min | `—` |
| `05_attn_small` | 0 | 0 | 213.6 | 217.7 | 412 | 247.3 | 214.3 min | `—` |
| `05_attn_small` | 1 | 0 | 250.0 | 250.4 | 318 | 274.0 | 165.2 min | `—` |
| `05_attn_small` | 2 | 0 | 219.0 | 229.5 | 292 | 266.3 | 143.0 min | `—` |
| `06_attn_mid` | 0 | 0 | 244.2 | 241.1 | 257 | 265.2 | 212.9 min | `—` |
| `06_attn_mid` | 1 | 0 | 248.0 | 245.3 | 261 | 266.7 | 229.4 min | `—` |
| `06_attn_mid` | 2 | 0 | 230.4 | 232.6 | 288 | 266.7 | 251.8 min | `—` |
| `07_resmlp_mid` | 0 | 0 | 241.3 | 237.5 | 470 | 268.9 | 115.1 min | `—` |
| `07_resmlp_mid` | 1 | 0 | 201.3 | 169.8 | never | 254.8 | 116.6 min | `—` |
| `07_resmlp_mid` | 2 | 0 | 242.8 | 237.9 | 528 | 276.3 | 119.5 min | `—` |
| `08_mlp_anchor` | 0 | 0 | 205.2 | 156.2 | never | 255.0 | 61.7 min | `—` |
| `08_mlp_anchor` | 1 | 0 | 33.0 | 25.7 | never | 59.8 | 39.6 min | `—` |
| `08_mlp_anchor` | 2 | 0 | 80.0 | 61.3 | never | 148.9 | 33.9 min | `—` |

## Failures

None.
