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

Runtime is 300 iterations x 5 episodes x seeds 0, 1, 2. Run it with the loop in
`AGENTS.md`, then regenerate the numbers with:

```bash
python manage.py report --sweep experiments/sweeps/cartpole_sweep_s0-2
```
