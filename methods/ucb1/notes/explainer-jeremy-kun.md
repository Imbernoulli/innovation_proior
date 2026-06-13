# Third-party explainer: Jeremy Kun, "Optimism in the Face of Uncertainty: the UCB1 Algorithm" (jeremykun.com, 2013)

URL: https://www.jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/
Captured via WebFetch (HTML explainer).

## Regret
Expected cumulative regret = mu* T - E[G_A(T)], where mu* is the best arm's mean and
G_A(T) is the algorithm's total reward over T rounds. Compare against the feasible best.

## Optimism principle (the core intuition)
Build an OPTIMISTIC estimate of each arm's value, then play the largest.
- If the optimistic estimate is WRONG (arm is actually bad), pulling it makes the bound
  drop quickly -> the algorithm moves on (exploration is self-correcting).
- If it is RIGHT, the algorithm keeps exploiting that arm.
This is why you add an UPPER confidence term rather than just the empirical mean:
the empirical mean alone is exploitation; the radius drives exploration of under-pulled arms.

## Index
Play j maximizing  x_bar_j + sqrt(2 log t / n_j).
- x_bar_j = empirical mean of arm j (exploitation term)
- sqrt(2 log t / n_j) = confidence radius (exploration term)

## Confidence radius derivation (Chernoff-Hoeffding)
For independent [0,1] r.v.s with empirical average Y over n samples and true mean mu:
  P(Y + a < mu) <= e^{-2 n a^2}.
Set the failure probability to t^{-4} with n = n_j:
  e^{-2 n_j a^2} = t^{-4}  =>  2 n_j a^2 = 4 log t  =>  a = sqrt(2 log t / n_j).
Two properties:
  (1) under-played arms accumulate a LARGER radius as t grows (log t numerator, fixed n_j) ->
      they are never permanently dismissed;
  (2) the radius SHRINKS as an arm is pulled (n_j grows) -> good arms get exploited.

## Regret bound
E[R_UCB1(T)] <= 8 sum_{i: Delta_i>0} (log T / Delta_i) + (1 + pi^2/3) sum_j Delta_j,
with Delta_i = mu* - mu_i. Gives O(sqrt(K T log T)) worst-case; better in practice.

## Exploration/exploitation balance
Empirical mean = exploitation; radius = exploration. As evidence accumulates the intervals
tighten, so emphasis shifts toward exploitation automatically.
