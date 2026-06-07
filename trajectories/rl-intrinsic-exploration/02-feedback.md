# Feedback after step 2 — ICM

Real leaderboard numbers (`baseline:icm`, `is_final,true`), seeds {42, 123, 456} and mean.

## Private Eye
| seed | eval_return | auc | nonzero_rate |
|---|---|---|---|
| 42 | 0.0 | −37.89 | 0.0 |
| 123 | 0.0 | −24.99 | 0.0 |
| 456 | 0.0 | −24.99 | 0.0 |
| **mean** | **0.0** | **−29.29** | **0.0** |

## Tutankham
| seed | eval_return | auc | nonzero_rate |
|---|---|---|---|
| 42 | 112.6 | 100.26 | 1.0 |
| 123 | 106.0 | 62.15 | 1.0 |
| 456 | 108.4 | 97.49 | 1.0 |
| **mean** | **109.0** | **86.63** | **1.0** |

## Frostbite
| seed | eval_return | auc | nonzero_rate |
|---|---|---|---|
| 42 | 232.0 | 1275.05 | 1.0 |
| 123 | 3066.0 | 1970.14 | 1.0 |
| 456 | 170.0 | 130.45 | 1.0 |
| **mean** | **1156.0** | **1125.21** | **1.0** |

## Reading the dynamics

Curiosity did exactly what step 1 needed on two of the three games, and revealed its limit on the
third.

**Tutankham is rescued and stabilized.** Where PPO had two dead seeds (0.0) and one lucky one, ICM
gets all three seeds to ~106–113 (`nonzero_rate` 1.0 everywhere, mean 109.0 vs PPO's 36.5). The
variance collapsed — the spread is now ~7 points instead of 0-vs-110. This is the clearest evidence
that the bonus converted exploration from a coin flip into something reliable: every seed now finds
and holds the reward. That is precisely the failure mode I diagnosed after step 1, and curiosity
fixed it.

**Frostbite jumped, but unevenly.** The mean `eval_return` rose from 215 (PPO) to **1156**, and the
`auc` numbers are large (1125 mean, with seed 42 at 1275 and seed 123 at 1970) — curiosity kept
pushing the agent into new controllable configurations and it climbed much higher than undirected
PPO. But look at the per-seed eval spread: {232, **3066**, 170}. One seed found a deep policy; the
other two landed near where PPO already was. The mean is carried almost entirely by seed 123. So the
Frostbite "win" is real but *fragile* — it's a high-variance jackpot, not a dependable lift. The
huge `auc` relative to final `eval_return` on seeds 42/456 also suggests curiosity drove strong
*early* exploration that didn't all consolidate into final return.

**Private Eye is untouched — and this is the diagnostic.** Every seed is exactly 0.0 with
`nonzero_rate` 0.0: ICM never finds a single Private Eye reward. Notice, though, that the `auc` is
now ~−29 and *bounded*, where PPO had a −535 seed and a −300 mean. So curiosity didn't *help* Private
Eye reach reward, but it did stop the agent from blundering into the big penalties — it makes the
agent explore its controllable surroundings rather than wander destructively. Why no reward, then?
Private Eye's payoff is far *and* the controllable dynamics near the start are quickly mastered; once
ICM's forward model has learned the local transitions, the bonus on that region *decays toward zero*,
and the agent loses its drive before it has crossed the long reward-free gap. This is the structural
weakness of a prediction-error bonus: it is a one-shot frontier-pusher that vanishes as the model
catches up, so it can't sustain the persistent, repeated traversal a game like Private Eye demands.

Net: ICM is a clear step up from PPO (two games improved, Private Eye made less harmful), but its
gains are seed-fragile on Frostbite and it cannot crack Private Eye at all. The next step needs a
*global* novelty signal that doesn't decay the moment local dynamics are learned — something that
keeps marking a state as "still unfamiliar overall," so exploration doesn't stall at the first
mastered region.
