# AGENTS.md

## Overview
Py-Gym is a config-driven PyTorch RL framework. `manage.py` is the only entrypoint.
Top-level packages (`agent`, `config`, `utils`) are imported directly, so run all
commands from the repository root. There is no build step or installed package layout.

## Setup
- Requires Python >= 3.11 (`tomllib`); the default `python` here may be 3.10.
- Install with `pip install -r requirements.txt`, or `poetry install` (README
  recommends running the Poetry install twice for PEP 517 build dependencies).
- For tests with pip, use `pip install -r requirements-dev.txt`.
- Box2D environments (`LunarLander-v3` is the reference one) need the system
  `swig` on PATH *before* the `box2d` extra is built (`sudo apt-get install -y
  swig`). The PyPI `swig` package does not satisfy build isolation. Without it,
  Box2D tests skip instead of failing.
- Training defaults to CPU; pass `--device cuda` when available.

## Commands
- Train: `python manage.py train config/templates/lunarlander_ppo.toml`
  - TOML and JSON are both supported; the file extension selects the loader.
  - `-T/--iterations` (default 500), `-N/--episodes` (episodes per iteration),
    `--eval N`, `--device cpu|cuda`.
  - `--seed N` makes a run reproducible. Omitting it draws a fresh seed, so
    every run stays unlinked; the seed actually used is always written to
    `config.json` / `config.toml` metadata. Bit-identical reruns also need the
    same `OMP_NUM_THREADS`, because torch CPU kernels may reorder reductions.
- Test a run: `python manage.py test experiments/<run_dir> -N 5`
  - Requires `config.toml` (or `config.json`) and `model.ckpt` in that directory.
- Headless: `--record cli` avoids opening the live matplotlib window
  (the default is `cli live`).
- Smoke test: `python manage.py train <config> --dry -T 1 -N 1`.
  `--debug` implies `--dry` and enables verbose logging.
- Sweeps: `poetry run python scripts/sweep.py --configs '<glob>' --seeds 0 1 2
  --jobs 4 --threads 4`
  - Runs each (config, seed) into `experiments/sweeps/<name>/<stem>/seed<N>/`
    through `--save-dir`, so parallel launches never collide on the timestamped
    default path.
  - Resumable: runs whose `run.json` reports `returncode: 0` are skipped;
    `--force` redoes them, `--plan-only` prints the matrix.
  - Ends with a `summary.csv` holding last-50/last-100, solved iteration, best
    and wall time per run.
  - `--save-dir DIR` also works on its own with plain `manage.py train`.
- Tests: `pytest` (tests live in `tests/` and are tracked).
  - `tests/test_templates_smoke.py` is marked `integration`: it builds every
    template and steps its env. Box2D cases skip when `Box2D` is missing.
  - `pytest -m "not integration"` runs the fast, dependency-light subset.
- There is no CI or lint/format/typecheck configuration; run pytest plus a
  short `--dry` training run as verification.

## Architecture
- `agent/learner/`: agent implementations. `BaseAgent.select_action()` owns
  numpy/tensor/device conversion, calls subclass `_act()`, then converts the raw
  action through `policy.handler.to_env_format()`. New agents must be exported in
  `agent/learner/__init__.py::__all__`.
- `agent/components/{encoder,policy,buffer}/`: composable modules. A policy owns
  `actor`, `critic` (optional), and `handler`.
- `utils/builder.py`: factory and component registry.
  - Registry is assembled from `__all__` of `agent`, the component packages,
    and `action_handler`. New classes must be exported there; a new component
    module must also be imported in `utils/builder.py`.
  - `build_component()` returns `(component, output_dim)`. It builds
    `sub_components()` recursively and threads the input dimension down, so
    actor/critic layers usually do not need an explicit `input_dim`. Components
    that declare `requires_input_dim = True` (e.g. `TransformerCritic`) receive
    the inherited dimension automatically.
  - `build_agent()` uses `agent_cls.sub_components()` as the component whitelist,
    rejects unknown `[agent.*]` keys, passes `hyper_params` as a dict, and
    resolves `hyper_params.criterion` into loss objects.
- Config structure: `[agent] type` plus flat `[agent.<component>]` tables.
  Do not use a `components` table.
- Dimensions: encoder output `feature_dim` feeds policy actor/critic. Configs
  must not define `state_dim`; it is inferred from `env.observation_space`.
- Action handling: `NormalHandler` / `BetaHandler` apply action bounds in
  `to_env_format()`; policy log-prob corrections include the scaling Jacobian.
- `ReinforceAgent.compute_returns()` resets the running return at every
  `dones` boundary, so a buffer holding several episodes is still correct.
- Seeding lives in `utils/seed.py`. `set_global_seed()` runs in `manage.py`
  *before* `build_agent()`, because weight initialisation consumes torch's
  global RNG. `episode_seed(seed, index)` derives env seeds as a pure function
  of the run seed and episode index, so runs sharing a seed see the same initial
  states no matter how many episodes they reset.

## Config and artifacts
- `env.default` is merged into both training and testing; `env.training` and
  `env.testing` override per mode.
- Config directories carry intent: `config/templates/` holds minimal runnable
  examples, `config/stable/` holds tuned, reproducible recipes, and
  `config/draft/` is gitignored scratch space for unproven ideas.
- A completed run writes `experiments/{env.id}_{timestamp}/` with `config.toml`,
  `config.json`, `model.ckpt` (including optimizer state), and
  `training_log.csv/png`. Ctrl-C intentionally skips config and checkpoint saving.
- Docs drift is possible; trust `manage.py`, the config templates, and the
  builder contract above.
