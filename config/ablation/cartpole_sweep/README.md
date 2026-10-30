# CartPole sweep

A deliberately cheap matrix (`CartPole-v1`, 32-feature encoder). It is a smoke
test for the experiment workflow, not a research question: every arm goes
through the same builder, completion contract, artifact split and report, and
the critic implementations moved under `agent/components/policy/` must train in
a second environment.

## Design

Everything except `[agent.policy.critic]` is identical across the three PPO
arms. `04_reinforce` drops the critic and the PPO-specific hyper-parameters.

| Arm | Agent | Critic |
| :--- | :--- | :--- |
| `01_ppo_mlp` | `PPOAgent` | `Sequential` (32 -> 64 -> 1) |
| `02_ppo_resmlp` | `PPOAgent` | `ResidualMLPCritic` (2 blocks, hidden 64) |
| `03_ppo_attn` | `PPOAgent` | `TransformerCritic` (4 tokens, 1 layer, 4 heads) |
| `04_reinforce` | `ReinforceAgent` | none |

## Budget

The configs fix nothing about the budget: `-T/--iterations` and `-N/--episodes`
are flags, so each sweep is named after what it actually ran and the record's
`config.toml` is the authority. Pass the same numbers to `report` in the figure
title, or the plot claims a budget its curves did not use.

| Sweep | Budget | What it is for |
| :--- | :--- | :--- |
| `cartpole_sweep_s0-2` | 300 x 5, seeds 0-2 | workflow smoke test: builder, completion contract, artifact split, report |
| `cartpole_sweep_s0-2_T600` | 600 x 5, seeds 0-2 | first attempt at the solved threshold: almost every run gets there, seed variance stays high |
| `cartpole_sweep_s0-2_T800` | 800 x 5, seeds 0-2 | longer budget, to see whether the spread narrows or the arms plateau |

Run one with the loop in `AGENTS.md`, then regenerate that sweep:

```bash
python manage.py report --sweep experiments/sweeps/cartpole_sweep_s0-2_T800 \
  --panel 'PPO critics:01_ppo_mlp,02_ppo_resmlp,03_ppo_attn' \
  --panel 'baseline:04_reinforce' \
  --title 'CartPole-v1, 800 iterations x 5 episodes x 3 seeds, mean +/- sd'
```

`summary.csv` holds the numbers, `REPORT.md` is generated from it, and each
sweep directory's `RESULTS.md` holds the interpretation.
