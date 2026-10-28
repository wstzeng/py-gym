# Development Roadmap

This roadmap follows parallel development **tracks** to evolve the Py-Gym Framework.

## Algorithm Track (Policy & Learning Logic)
- [x] **Vanilla PG**: REINFORCE implementation.
- [x] **Actor-Critic**: Advantage-based actor-critic.
- [x] **PPO**: Proximal Policy Optimization with GAE.
- [ ] **Off-Policy**: SAC or TD3 for sample efficiency.

## Architecture Track (System Design)
- [x] **OOP Foundation**: Base classes and module hierarchy.
- [x] **Object Factory**: Config-driven builder logic.
- [x] **Agnostic Interface I**: Unified Pipeline in BaseAgent.
- [x] **Agnostic Interface II**: Space-agnostic (Discrete/Continuous) support.
- [x] **Config System**: Hyper-parameter management via Dataclasses.
- [ ] **Visual Encoders**: CNN support for pixel-based states.
- [ ] **Generalized Encoders**: Introduce more architecture support.
- [ ] **MARL**: Architecture support for Multi-Agent systems.

## Experiment Track (Tooling & Artifacts)
- [x] **Monitoring**: Real-time CLI and Live monitors.
- [x] **Unified Artifacts**: Integrated saving of configs, logs, and ckpt.
- [ ] **Adaptive Visualizer**: (New) Multi-mode rendering (ASCII-CLI, Matplotlib-GUI, No-Plot).
- [ ] **Reporting**: Auto-generation of high-res training reports (PDF/PNG) with Z-order trend analysis.
- [ ] **Lab Manager**: A dedicated tool/CLI for experiment archiving and batch comparison.
- [ ] **Web Dashboard**: A lightweight web interface (Streamlit or Gradio based) to browse `experiments/` folders and compare curves dynamically.

## Methodology Track (Trustworthy Comparisons)

Gaps found while running the LunarLander PPO / TransformerCritic / REINFORCE
comparison. The core abstractions held up; what is missing is the layer that
makes an experiment result *believable*.

- [x] **Seeding**: Global seed control (env, action sampling, weight init, optimizer)
  so runs are reproducible and A/B comparisons are meaningful. Implemented as
  `--seed` plus `utils/seed.py`; the resolved seed is recorded in run metadata.
- [ ] **Best Checkpoint**: Save `model.ckpt` whenever the trailing-100 mean reward
  improves, not only after the final iteration. A run that peaks at iteration 720
  and decays currently keeps only the decayed weights.
- [ ] **Run Isolation**: Disambiguate `experiments/{env.id}_{timestamp}/` (append the
  pid or a short hash). Two runs of the same env started within the same second
  currently share a directory, and `exist_ok=True` silently overwrites.
- [ ] **Resume**: `--resume <run_dir>` to continue from a saved checkpoint and
  optimizer state. Optimizer state is already persisted but cannot be consumed.
- [ ] **Step Budget**: Optional fixed environment-step budget per iteration instead of
  a fixed episode count, so per-iteration cost and wall time stay comparable across
  configs. Episode length growth is why a run is ~4x slower at iteration 350 than at 50.
- [ ] **Raw Metrics**: Record unweighted values (entropy, critic loss) next to the
  tracker-weighted ones. `training_log.csv` currently stores `entropy * -entropy_weight`,
  so the raw number is unrecoverable without knowing the weight.
- [ ] **In-Training Evaluation**: Periodic deterministic evaluation written to a second
  CSV. The reported `AverageReward` is sampled and therefore not comparable to a
  test-time deterministic score.

### Suggested order

| Priority | Item | Effort |
| ---: | :--- | :--- |
| 1 | Best Checkpoint | 0.5 day |
| 2 | Run Isolation | 1 hour |
| 3 | Raw Metrics | 1 hour |
| 4 | Resume | 1 day |
| 5 | Step Budget | 2 days |
| 6 | In-Training Evaluation | 1 day |

Items 1–3 fit in a single day and are the foundation for every future
"is this config actually better?" question.

