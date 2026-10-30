# CartPole sweep at 800 iterations — what this run was for

The 600-iteration sweep left one run below the solved threshold and every arm
still climbing, so the question here was narrow: does a longer budget converge,
or does each arm plateau where it already was?

## What it establishes

- **The budget was the binding constraint, not the method.** All twelve runs now
  reach the solved threshold, including the residual-MLP seed that missed it at
  600. The threshold is reached late — well past the halfway point — so the
  earlier sweep was measuring "not enough steps" rather than "this critic does
  not learn".
- **Nothing has converged yet.** Averaging the last 300 iterations against the
  last 100 still shows every arm improving, so the ranking is a snapshot of a
  rising curve, not of a settled policy.
- **The spread narrowed for the arms that were stable, and not for the ones that
  were not.** The transformer and REINFORCE arms now agree closely across seeds;
  the plain and residual MLP critics still differ by tens of points between
  seeds, so their means are the least trustworthy numbers in the table.
- **The critic does not pay for itself at this scale.** The critic-free baseline
  matches or beats all three PPO arms. With 32 encoder features, a shared
  learning rate and 5 episodes per iteration there is very little for a value
  function to model, so this says nothing about critics on a harder
  environment — only that CartPole cannot rank them.

## What it does not establish

Wall time per run grows across the arms (transformer is the most expensive) and
the budget is shared rather than tuned per arm, so the arms are not compared at
equal cost. Three seeds cannot separate two arms whose intervals overlap; treat
`summary.csv` as the numbers, this file as the caveat, and the comparison as
"does the workflow hold", not "which critic wins".
