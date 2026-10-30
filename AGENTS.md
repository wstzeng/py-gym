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
  - The loader picks by extension; hand-written configs are TOML, and legacy
    `.json` is read but never written (see "Config and artifacts").
  - `-T/--iterations` (default 500), `-N/--episodes` (episodes per iteration),
    `--eval N`, `--device cpu|cuda`.
  - `--seed N` makes a run reproducible. Omitting it draws a fresh seed, so
    every run stays unlinked; the seed actually used is always written to
    `config.toml` metadata. Bit-identical reruns also need the same
    `OMP_NUM_THREADS`, because torch CPU kernels may reorder reductions.
- Test a run: `python manage.py test experiments/<record> -N 5`
  - Requires `config.toml`; the checkpoint is looked up at the mirrored path
    under `artifacts/`, not next to the record.
- Headless: `--record cli` avoids opening the live matplotlib window
  (the default is `cli live`).
- Smoke test: `python manage.py train <config> --dry -T 1 -N 1`.
  `--debug` implies `--dry` and enables verbose logging.
- Matrix runs: there is no sweep command; a shell loop over the matrix is the
  whole tool. It skips records that already exist, so a re-run is a resume:

  ```bash
  for cfg in config/ablation/*.toml; do for seed in 0 1 2; do
    out=experiments/sweeps/<name>/$(basename "$cfg" .toml)/seed$seed
    [ -e "$out/config.toml" ] && continue
    OMP_NUM_THREADS=4 python manage.py train "$cfg" -T 600 -N 20 --seed "$seed" \
      --save-dir "$out" --record cli
  done; done
  ```

  Parallelise by wrapping the body in `xargs -P 4`; give each worker the same
  thread count so the recorded provenance stays comparable.
- Report a sweep: `python manage.py report --sweep experiments/sweeps/<name>`
  rebuilds `summary.csv`, `REPORT.md`, `comparison.png` and every run's
  `training_log.png` from the logs alone (`--panel 'Title:arm1,arm2'` splits
  tiers, `--smooth`, `--threshold`, `--no-run-plots`). One run:
  `manage.py report --run <record>`.
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
- **Where a component goes is decided by the slot it fills, never by the builder
  mechanism it uses.** `requires_input_dim`, `sub_components()` and `feature_dim`
  are shared plumbing; "it needs an input dim and contains a network" is not a
  reason to put something under `encoder/`. Get this wrong and nothing fails:
  the registry is one flat namespace, so a value network dropped into `encoder/`
  still builds, still trains, and silently returns a scalar where the policy
  expects features.
  - `encoder/`: observation -> latent features. Every export must subclass
    `BaseEncoder` and expose `feature_dim`; `build_agent()` threads that number
    into the policy. No value networks.
  - `policy/`: the policy classes plus the leaf networks of their sub-slots —
    `actor_*.py`, `*_critic.py` (one value per sample, shape `(N, 1)`),
    `action_handler.py`. These are graph leaves, not policies; they are exported
    from `policy/__init__.py::__all__` so the registry sees them.
  - Add a sub-slot implementation as a module inside the owning package. Only
    create a new package for a new **agent-level** slot, i.e. a new name in
    `BaseAgent.sub_components()`.
  - `tests/test_component_placement.py` enforces the rule; `pytest` will not
    catch a misplacement any other way.
- Class names are part of the config format (`type = "ConvCritic"`) and every
  recorded run keeps its `config.toml` forever. Moving a class between modules is
  safe — checkpoints key on attribute paths such as `policy.critic.project.weight`.
  Renaming a class is a breaking change to all recorded runs and needs a
  compatibility export.
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
- One format is written: TOML. `AgentConfig.save()` dumps to a sibling `.tmp`
  and `os.replace()`s it into place; loading reads `.toml` first and falls back
  to a legacy `.json` snapshot, so recorded runs from before this rule still
  open. `json5` is gone; there is no second writer to keep in sync.
- Unknown top-level sections are rejected, so a typo cannot silently drop a
  whole table.
- A run is complete when its record directory holds `config.toml`. The snapshot
  is written last, after the checkpoint and the CSV, so an interrupted run
  leaves none; `train` then clears the known leftovers and starts from
  iteration 0, and refuses to overwrite a directory that already has one.
- Records and artifacts are two trees with one layout. `experiments/` is
  tracked and branch-scoped: `config.toml`, `training_log.csv`, and generated
  `summary.csv` / `REPORT.md` / `comparison.png`, plus a `RESULTS.md` for
  interpretation. `artifacts/` is ignored and mirrors the same relative paths;
  it holds `model.ckpt`, `training_log.png` and `train.log`. `paths.artifact_for()`
  maps one to the other, and a `--save-dir` outside `experiments/` (a dry run)
  keeps everything together. The plot is drawn by `report` from the CSV, so
  deleting `artifacts/` costs pictures and checkpoints, never numbers.
- Config directories carry intent: `config/templates/` holds minimal runnable
  examples, `config/stable/` holds tuned, reproducible recipes,
  `config/ablation/` holds deliberately designed comparison matrices, and
  `config/draft/` is gitignored scratch space for unproven ideas.
- Config directories hold only `.toml` (plus, at most, a README describing the
  experiment's design). Results do **not** live there.
- Measured numbers belong in generated files, not prose: `summary.csv` is the
  source of truth, `REPORT.md` is generated from it, and `comparison.png` is
  regenerated from `training_log.csv`. `RESULTS.md` holds interpretation only.
- Docs drift is possible; trust `manage.py`, the config templates, and the
  builder contract above.

## Branches and experiment records
- Code lives on `feat/*`, configurations and their results on `exp/*`. A feature
  branch merges into `main` with `--no-ff`. **Never rebase an `exp/*` branch**:
  its result commits are history, and rewriting them to move the base buys
  nothing while forcing a force-push.
- An `exp/*` branch only ever takes a **tagged** commit of `main`, never the
  branch tip: start it at a release tag (`git switch -c exp/<name> v0.2.0`) and
  later merge the next tag (`git merge --no-ff v0.2.0`, which an annotated tag
  records as `Merge tag 'v0.2.0'`). Untagged code therefore reaches an
  experiment only through a release, so the `git_sha` every run records is
  always a published version — records stay reproducible and comparable across
  branches. The cost is that refactors wait for a release, so tag whenever a
  change lands that experiments should be able to pick up.
- Releases are annotated `vMAJOR.MINOR.PATCH` tags on `main`, cut from a commit
  whose `pyproject.toml` already carries that number — bump the version first,
  verify, then tag, so the tag never names a tree that disagrees with it.
- One checkout, no worktrees. `experiments/` is tracked on `exp/*`, so switching
  branch switches which records are on disk while `artifacts/` stays put.
  Because a run writes into a branch's tree, **do not switch branches while a
  matrix is running**: finish or stop it, merge, then move on.
- Every run records its provenance (`utils/provenance.py`): `git_sha`,
  `git_dirty`, `torch_threads`, `omp_num_threads` and the python/torch/numpy
  versions. `git_dirty: true` means the working tree did not match that commit.
