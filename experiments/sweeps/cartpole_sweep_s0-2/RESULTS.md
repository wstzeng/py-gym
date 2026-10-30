# CartPole sweep — what this run was for

This matrix is a workflow test, not a critic comparison. Its value is in what it
exercises: four arms through one builder, one completion contract, one artifact
split and one report, on an environment that has nothing to do with LunarLander.

## What it establishes

- **The refactor works.** Twelve runs completed, each record holds only
  `config.toml` and `training_log.csv`, and every checkpoint and plot landed in
  the mirrored path under `artifacts/`. `manage.py test` loads one of them
  through that mirror without being told where the checkpoint is.
- **It is reproducible.** Rerunning `01_ppo_mlp` seed 0 with the same seed and
  `OMP_NUM_THREADS=2` produced a byte-identical `training_log.csv`.
- **The contracts hold.** Re-running into a finished directory is refused, and
  the recorded `git_sha` is the commit that introduced the matrix, flagged clean
  even while the untracked result directories were being written.
- **It is portable.** The three moved critic implementations and the REINFORCE
  baseline all train on a discrete-action Box2D-free environment, so nothing in
  the component placement depends on the LunarLander setup.

## What it does not establish

No arm reaches the solved threshold within 300 iterations x 5 episodes, so the
ordering between arms says nothing about the critics. The transformer arm leads
and the residual-MLP arm trails, but the budget is small, the hyper-parameters
are shared rather than tuned, and every arm still has high seed variance. Treat
`summary.csv` as the numbers and this file as the caveat.
