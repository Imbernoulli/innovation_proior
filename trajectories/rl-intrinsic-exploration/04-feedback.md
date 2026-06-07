# Feedback after step 4 (finale) — NGU

There is no NGU row in this task's `leaderboard.csv` — the leaderboard contains the three baseline
methods (`baseline:ppo`, `baseline:rnd`, `baseline:icm`) plus agent runs, but no measured NGU result.
So I do **not** invent numbers here. Instead I state the bar it has to clear (the real strongest-
baseline numbers) and exactly what I would validate, framed as the hypotheses the two-timescale design
makes falsifiable.

## The bar to beat (real numbers, `baseline:rnd`, the strongest baseline)

| game | metric | RND mean | RND per-seed |
|---|---|---|---|
| Private Eye | eval_return | 952.0 | {0, 100, 2756} |
| Private Eye | auc | −84.23 | {−260.65, −98.47, 106.43} |
| Private Eye | nonzero_rate | 0.667 | {0, 1, 1} |
| Tutankham | eval_return | 100.33 | {116.2, 102.2, 82.6} |
| Frostbite | eval_return | 250.67 | {216, 250, 286} |

The diagnosis the finale is built on is grounded in those numbers: RND is the only baseline to move
Private Eye at all (the 2756 seed), but its Private Eye `auc` mean is **negative** (−84.23), one of
three seeds is a flat zero (`nonzero_rate` 0.667), and seed 42's `auc` of −260.65 says it spent
training in the red. That is the precise signature of a *lifelong-only* bonus with no within-episode
reset: it can reach the reward on a lucky seed but cannot reliably re-traverse cleared ground to do so
again.

## What I would validate

1. **Private Eye reliability, not just a lucky seed.** The whole point of the episodic reset is
   persistence, so the falsifiable claim is on the *worst* seed and on `auc`, not the headline eval.
   I would check whether `nonzero_rate` moves off 0.667 toward 1.0 (the dead seed 42 stops finding
   nothing) and whether the Private Eye `auc` lifts from negative toward positive (less time in the
   red, more sustained progress). If the episodic timescale is the right fix, those two numbers move
   *before* the peak eval does.
2. **No regression on the easier games.** Tutankham and Frostbite are where the bonus matters less
   once reward is reachable; the UVFA exploit head ($\beta_0=0$) exists precisely so a non-vanishing
   bonus doesn't drag exploitation. I would confirm Tutankham stays ~100 (RND) / ~109 (ICM) and that
   Frostbite holds near RND's stable ~250 — and watch whether the episodic drive recovers any of
   ICM's Frostbite upside (its 3066 seed) without ICM's variance.
3. **Variance across seeds shrinks.** The recurring story up the ladder is high seed-to-seed variance
   (PPO's 1100-point Private Eye swing, ICM's one-seed Frostbite jackpot, RND's dead Private Eye
   seed). The two-timescale bonus is supposed to make the *worst* seed better, so I would track the
   per-seed spread, not just the mean.
4. **The episodic memory actually pays its compute.** The kNN lookup over a per-episode memory is the
   one real cost added on top of RND; I would confirm the `elapsed_*` cost stays within the same
   budget the baselines used and that the memory's ring-buffer cap keeps it bounded.

The honest summary: NGU is the natural next step from RND's exact failure mode — it keeps RND's
global signal verbatim and adds the one thing RND structurally lacks, a within-episode novelty that
resets each episode so the agent never gives up re-exploring. Whether that converts RND's fragile
Private Eye breakthrough into a dependable one is the empirical question this trajectory ends on; the
numbers above are the bar, and the four checks above are how I would read the answer.
