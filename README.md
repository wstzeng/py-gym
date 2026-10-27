# Py-Gym Framework

An object-oriented reinforcement learning framework focused on modularity and clear separation of concerns.

## Project Goals

- **Architecture Scalability**: Leveraging OOP principles to build a scalable hierarchy for diverse RL algorithms.
- **Reproducibility**: Automatic experiment tracking with integrated configs, logs, and checkpoints.
- **Algorithmic Clarity**: Implementing core RL methods with readable, well-structured code.

## Architecture & Module Responsibilities

The framework follows a strict modular design:

- **[Agent](agent/)**: Orchestrates decision-making and learning.
    - **[Learner](agent/learner)**: Agent implementations and algorithm-specific update logic.
    - **[Encoder](agent/components/encoder)**: State representation learning and feature extraction.
    - **[Policy](agent/components/policy)**: Distribution, value, and action-handler logic.
    - **[Buffer](agent/components/buffer)**: Rollout and experience storage.
- **[Config](config/)**: TOML/JSON configuration entities.
- **[Utils](utils/)**: Builders, training loops, logging, and experiment artifacts.

State dimensions are inferred at build time: the encoder receives the environment
observation size, and its output dimension is threaded into the policy's actor and
critic networks.

## Configuration System

The framework is **configuration-driven**. Each component declares its own
`type` and optional `params`; nested components are declared directly under their
parent instead of inside a `components` table:

```toml
[agent]
type = "PPOAgent"

[agent.encoder]
type = "VectorEncoder"
layers = [{ type = "Linear", out_features = 64 }]

[agent.policy]
type = "ActorCriticPolicy"

[agent.policy.actor]
type = "Sequential"
layers = [{ type = "Linear", out_features = 2 }]
```

See the **[Configuration Specification](config/Config.md)** and the example
recipes in **[config/templates](config/templates)**.

## Experiment Management

Every training run generates a self-contained directory in `experiments/`:

```text
experiments/ENV_TIMESTAMP/
├── config.toml       # Full configuration (TOML)
├── config.json       # Same contents as `config.toml` in JSON format
├── model.ckpt        # Weights and optimizer state
├── training_log.csv  # Step-by-step metrics (Reward, Loss)
└── training_log.png  # Visualized performance curve
```

## Setup & Usage

### 1. Installation

Python 3.11 or newer is required.

> **Note**: The dual-stage Poetry installation is a workaround for packages that
> do not fully comply with PEP 517. Running it twice ensures build dependencies
> are available before legacy build backends run.

#### Using Poetry

```bash
poetry install
poetry run poetry install
```

#### Using standard pip

```bash
pip install -r requirements.txt
```

For the test suite, install the development requirements:

```bash
pip install -r requirements-dev.txt
```

### 2. Basic Commands

Train with a configuration:

```bash
python manage.py train config/templates/lunarlander_ppo.toml
```

`toml` and `json` configs are both supported. Useful flags:

```bash
python manage.py train config/templates/cartpole_ppo.toml \
    -T 100 -N 5 --device cpu --record cli
```

- `-T/--iterations`: training iterations (default 500).
- `-N/--episodes`: episodes per iteration (default 10).
- `--eval N`: run N evaluation episodes after training.
- `--record cli live`: output modes (default `cli live`); use `cli` on headless machines.
- `--dry`: write artifacts to a temporary directory.
- `--debug`: verbose logging; also implies `--dry`.

Test a trained model:

```bash
python manage.py test experiments/<experiment_directory>
```

Run the tests:

```bash
pytest
```

## Project Status

For the latest development progress and future plans, see the **[Roadmap](Roadmap.md)**.
