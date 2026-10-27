# AGENTS.md

## Overview
Py-Gym is a config-driven PyTorch RL framework. `manage.py` is the only entrypoint.
Top-level packages (`agent`, `config`, `utils`) are imported directly, so run all
commands from the repository root. There is no build step or installed package layout.

## Setup
- Requires Python >= 3.11 (`tomllib`); the default `python` here may be 3.10.
- Install with `pip install -r requirements.txt`, or `poetry install` (README
  recommends running the Poetry install twice for PEP 517 build dependencies).
- Training defaults to CPU; pass `--device cuda` when available.

## Commands
- Train: `python manage.py train config/templates/cartpole_ppo.toml`
  - TOML and JSON are both supported; the file extension selects the loader.
  - `-T/--iterations` (default 500), `-N/--episodes` (episodes per iteration),
    `--eval N`, `--device cpu|cuda`.
- Test a run: `python manage.py test experiments/<run_dir> -N 5`
  - Requires `config.toml` (or `config.json`) and `model.ckpt` in that directory.
- Headless: `--record cli` avoids opening the live matplotlib window
  (the default is `cli live`).
- Smoke test: `python manage.py train <config> --dry -T 1 -N 1`.
  `--debug` only raises the log level.
- Do not use the Poetry `train`/`test` scripts: `pyproject.toml` points them at
  `utils.train:main` and `utils.test:main`, which do not exist.
- There is no tracked test suite or CI. `pyproject.toml` declares pytest/black/ruff
  dev dependencies but configures none of them; verify changes with a short
  `--dry` run.

## Architecture
- `agent/entity/`: agent implementations. `BaseAgent.select_action()` owns
  numpy/tensor/device conversion, calls subclass `_acting()`, then
  `_to_env_action()`. New agents must be exported in
  `agent/entity/__init__.py::__all__`.
- `agent/{encoder,policy,buffer}/`: composable modules. A policy owns `actor`,
  `critic` (optional), and `handler`.
- `utils/builder.py`: factory and component registry. The registry is assembled
  from `__all__` of `agent`, `agent.encoder`, `agent.policy`,
  `agent.policy.action_handler`, and `agent.buffer`. New classes must be exported
  there; a new component module must also be imported in `utils/builder.py`.
- Config structure: `[agent] type` plus `[agent.components.*]`. `type` selects
  the class, `params` are constructor kwargs, and nested `components` tables are
  injected as kwargs. `layers` builds `nn.Sequential`, infers
  `Linear.in_features`, and injects `feature_dim`.
- `policy.feature_dim` must match the encoder output dimension; downstream
  actor/critic layers do not infer it.
- `build_agent()` infers state dimension from `env.observation_space` at runtime;
  the `env.state_dim` config value is not used.
- Known issue: `build_agent()` indexes `env.action_space.shape[0]` although
  `action_dim` is unused, so Discrete-action environments (including the CartPole
  template) fail; prefer continuous-action configs or patch that line.

## Config and artifacts
- `env.default` is merged into both training and testing; `env.training` and
  `env.testing` override per mode.
- Example configs live in `config/templates/`; add new recipes there.
- A completed run writes `experiments/{env.id}_{timestamp}/` with `config.toml`,
  `config.json`, `model.ckpt`, and `training_log.csv/png`. Ctrl-C intentionally
  skips config and checkpoint saving.
- Docs drift: README links `configs/Config.md` (actual path is `config/Config.md`)
  and shows only JSON; trust `manage.py` and the config templates.
