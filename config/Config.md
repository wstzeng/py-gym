# Configuration Specification

This project uses a component-based configuration system.
Each component is instantiated via the [`builder.py`](../utils/builder.py) using the following schema:
- `type`: The class name of the component.
- `params`: Arguments passed to the class constructor.
- `components`: Nested sub-components to be instantiated and injected.

---

## 1. Environment (`env`)
Defines the Gymnasium environment settings.

| Field | Description |
| :--- | :--- |
| `id` | Environment ID (e.g., "CartPole-v1"). |
| `state_dim` | Input dimension for the encoder. |
| `default` | General environment settings. |
| `training` | Settings specifically for the training environment. |
| `testing` | Settings specifically for the evaluation environment (e.g., `render_mode`). |

---

## 2. Agent ([`agent`](../agent))
The high-level agent structure.

### Components
- **[`encoder`](../agent/encoder)**: Transforms raw states into latent features.
    - `layers`: A list of layer definitions (Linear, ReLU, etc.) directly under the component.
- **[`policy`](../agent/policy)**: Reasoning core. Contains `actor`, `critic` (optional), and `handler`.
    - **[`handler`](../agent/policy/action_handler.py)**: Standardizes action space logic.
        | Handler Type | Required `params` | Actor Output Dim Requirement |
        | :--- | :--- | :--- |
        | `CategoricalHandler` | `action_dim` | `action_dim` |
        | `NormalHandler` | `action_dim`, `action_low`, `action_high` | `action_dim` |
        | `BetaHandler` | `action_dim`, `action_low`, `action_high` | `action_dim * 2` |
- **[`buffer`](../agent/buffer)**: Storage for experiences (e.g., `TrajectoryBuffer`).

---

## 3. Hyper Parameters (`hyper_params`)
Global training settings and optimization.

### Optimizer
Matches the standard component schema:
- `type`: Optimizer class (e.g., "AdamW", "Adam").
- `params`:
    - `lrs`: Dictionary mapping component/module names to learning rates.
        - Must include a `default` key.
        - Optional: specific keys (e.g., `critic`, `encoder`) to override default.
    - `weight_decay`: L2 regularization coefficient.
    - (Other standard PyTorch optimizer arguments like `betas`, `eps`).

### Algorithm Specifics
- `gamma`: Discount factor for future rewards.
- `entropy_weight`: Weight for the entropy bonus to encourage exploration.
- `eps_clip`: (PPO only) Clipping range for the objective function.
- `k_epochs`: (PPO only) Number of update iterations per batch.
- `critic_weight`: Importance weight for the value loss in Actor-Critic setups.
- `criterion`: Dictionary specifying loss functions (e.g., `"critic_loss": "MSELoss"`).
