# Py-Gym Framework

An object-oriented reinforcement learning framework focused on modularity and clear separation of concerns.

## Project Goals

- **Architecture Scalability**: Leveraging OOP principles to build a scalable hierarchy for diverse RL algorithms.
- **Reproducibility**: Automatic experiment tracking with integrated configs, logs, and checkpoints.
- **Algorithmic Clarity**: Implementing core RL methods with readable, well-structured code.

## Architecture & Module Responsibilities

The framework follows a strict modular design:

- **[Agent](agent/)**: Orchestrates decision-making and learning.
    - **[Policy](agent/policy)**: Neural network architectures and action selection logic.
    - **[Buffer](agent/buffer)**: Experience replay or rollout storage management.
    - **[Encoder](agent/encoder)**: State representation learning and feature extraction.
- **[Utils](utils/)**: Decoupled training logic, logging systems, and object builders.

## Experiment Management

Every training run generates a self-contained directory in `experiments/`:

```text
experiments/ENV_TIMESTAMP/
├── config.json       # Full configuration (arch + runtime params)
├── model.ckpt        # Trained weights (nn.Module state_dict)
├── training_log.csv  # Step-by-step metrics (Reward, Loss)
└── training_log.png  # Visualized performance curve
```

## Setup & Usage

```bash
# Install dependencies
poetry install

# Train an agent
poetry run python run_train.py experiments/lunar/ppo.json --iterations 500 --episodes 10

# Test a trained model
poetry run python run_test.py experiments/ENV_TIMESTAMP/
```

## Project Status

For the latest development progress and future plans, please see the **[Roadmap](Roadmap.md)**.
