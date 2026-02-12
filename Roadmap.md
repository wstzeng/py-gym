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
- [ ] **Agnostic Interface**: (New) Decorator-based Tensor conversion & Space-agnostic (Discrete/Continuous) support.
- [ ] **Visual Encoders**: CNN support for pixel-based states.
- [ ] **Generalized Encoders**: Introduce more architecture support.
- [ ] **MARL**: Architecture support for Multi-Agent systems.

## Experiment Track (Tooling & Artifacts)
- [x] **Monitoring**: Real-time CLI and Live monitors.
- [ ] **Unified Artifacts**: Integrated saving of configs, logs, and ckpt. (In Progress)
- [ ] **Adaptive Visualizer**: (New) Multi-mode rendering (ASCII-CLI, Matplotlib-GUI, No-Plot).
- [ ] **Reporting**: Auto-generation of high-res training reports (PDF/PNG) with Z-order trend analysis.
- [ ] **Lab Manager**: A dedicated tool/CLI for experiment archiving and batch comparison.
- [ ] **Web Dashboard**: A lightweight web interface (Streamlit or Gradio based) to browse `experiments/` folders and compare curves dynamically.
