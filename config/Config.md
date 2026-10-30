# Configuration Specification

This project uses a component-based configuration system. Components are
instantiated by [`utils/builder.py`](../utils/builder.py) from the following
schema:

A file has exactly four top-level sections: `env`, `agent`, `hyper_params` and
`metadata` (the last is written into run snapshots by `manage.py`). Any other
top-level key is an error rather than a silently ignored table.

- `type`: The class name of the component.
- `params`: Arguments passed to the class constructor.
- Sub-components: Declared directly under their parent, for example
  `[agent.policy.actor]` or `[agent.policy.handler]`.

The builder discovers component classes from the `__all__` exports of
`agent`, `agent.components.encoder`, `agent.components.policy`,
`agent.components.policy.action_handler`, and `agent.components.buffer`.

---

## 1. Environment (`env`)

Defines the Gymnasium environment settings.

| Field | Description |
| :--- | :--- |
| `id` | Environment ID (e.g., `"CartPole-v1"`). |
| `default` | General environment settings. |
| `training` | Settings specifically for the training environment. |
| `testing` | Settings specifically for evaluation (e.g., `render_mode`). |

`env.default` is merged into both training and testing; `env.training` and
`env.testing` add or override values for their respective modes.

The state dimension is inferred at build time from
`env.observation_space`; do not put `state_dim` in the config.

---

## 2. Agent (`agent`)

The high-level agent is selected with `[agent] type`. Its components are
declared directly under `[agent]`, one table per component.

```toml
[agent]
type = "PPOAgent"

[agent.encoder]
type = "VectorEncoder"
layers = [
    { type = "Linear", out_features = 64 },
    { type = "LeakyReLU" },
]

[agent.policy]
type = "ActorCriticPolicy"

[agent.policy.handler]
type = "NormalHandler"

[agent.policy.handler.params]
action_dim = 2
action_low = -1.0
action_high = 1.0

[agent.policy.actor]
type = "Sequential"
layers = [{ type = "Linear", out_features = 2 }]

[agent.policy.critic]
type = "Sequential"
layers = [
    { type = "Linear", out_features = 32 },
    { type = "LeakyReLU" },
    { type = "Linear", out_features = 1 },
]

[agent.buffer]
type = "TrajectoryBuffer"
```

### Components

- **`encoder`**: Transforms raw states into latent features.
    - `layers`: Layer definitions for `VectorEncoder`.
    - The encoder's output `feature_dim` is passed to the policy automatically.
    - Encoders produce features, never values. A value network placed here is
      accepted by the builder but threads nothing downstream — put it under
      `policy.critic` instead.
- **`policy`**: Distribution and value heads.
    - `actor`: Network producing distribution parameters.
    - `critic`: Optional value network for actor-critic algorithms. Either a
      plain `Sequential`, or one of the built-in variants in
      [`agent/components/policy`](../agent/components/policy):

        | Type | Mixing | Params |
        | :--- | :--- | :--- |
        | `ConvCritic` | local (1D conv over feature tokens) | `d_model`, `depth`, `n_tokens`, `kernel_size` |
        | `ResidualMLPCritic` | dense, LayerNorm + residual | `d_model`, `hidden`, `depth` |
        | `TransformerCritic` | self-attention over feature tokens | `d_model`, `nhead`, `depth`, `n_tokens`, `dim_feedforward` |

        All three take the encoder's `feature_dim` automatically and return one
        value per sample.
    - `handler`: Converts policy outputs into environment actions.
- **`buffer`**: Rollout storage (currently `TrajectoryBuffer`).

### Action Handlers

Handler implementations live in
[`agent/components/policy/action_handler.py`](../agent/components/policy/action_handler.py).

| Handler Type | Required `params` | Actor Output Dim Requirement |
| :--- | :--- | :--- |
| `CategoricalHandler` | `action_dim` | `action_dim` |
| `NormalHandler` | `action_dim`, `action_low`, `action_high` | `action_dim` |
| `BetaHandler` | `action_dim`, `action_low`, `action_high` | `action_dim * 2` |

### Dimension Flow

- `Linear` layers infer `in_features` from the component's `input_dim` (or from
  the parent component).
- Encode output feeds the policy; actor and critic receive the encoder's
  `feature_dim` automatically.
- If a component needs a different input dimension, set `input_dim` explicitly
  on that component.
- `feature_dim` is auto-detected for network-backed components but can be
  overridden in config.

---

## 3. Hyper Parameters (`hyper_params`)

Global training settings and optimization.

### Optimizer

```toml
[hyper_params.optimizer]
type = "AdamW"

[hyper_params.optimizer.params]
weight_decay = 0.01
lrs = { default = 0.001, actor = 0.0001 }
```

- `type`: Optimizer class from `torch.optim`.
- `lrs`: Learning rates keyed by module name. `default` is required when using
  per-module overrides; named keys (e.g., `actor`, `critic`, `encoder`) receive
  their own parameter group.
- Other keys are forwarded to the optimizer constructor.

### Algorithm Specifics

- `gamma`: Discount factor for future rewards.
- `entropy_weight`: Weight for the entropy bonus.
- `eps_clip`: (PPO only) Clipping range for the objective.
- `k_epochs`: (PPO only) Update iterations per batch.
- `critic_weight`: Weight for the value loss.
- `gae_lambda`: (PPO / actor-critic) GAE smoothing parameter.
- `criterion`: Loss-function mapping, for example
  `[hyper_params.criterion] critic_loss = "MSELoss"`. Values are resolved from
  `torch.nn` and passed to the agent as keyword arguments.
