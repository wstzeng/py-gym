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

## Configuration System

This framework is entirely **configuration-driven**. You can define environment settings, neural network architectures, and action handlers without modifying the source code.

For detailed documentation on how to write or customize your own configurations, please refer to the **[Configuration Specification](configs/Config.md)**.

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

### 1. Installation
Choose the method that fits your environment:

> **Note**: This dual-stage installation serves as a workaround for packages that do not fully comply with PEP 517.
> Running the installation twice ensures that build-time dependencies are correctly localized and available in the environment path before the compilation of legacy build backends begins.
#### Using Poetry
```bash
poetry install; poetry run poetry install
```

#### Using standard pip
```bash
pip install -r requirements.txt
```

### 2. Basic Commands

#### Train with configuration
```bash
python manage.py train <config_name>.json
```

#### Test a trained model
```bash
python manage.py test <experiment_directory>
```

## Project Status

For the latest development progress and future plans, please see the **[Roadmap](Roadmap.md)**.
