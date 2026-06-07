# Feedback after step 1 — vanilla PPO (no bonus)

Real leaderboard numbers (`baseline:ppo`, `is_final,true`), three seeds {42, 123, 456} and their mean,
across all three metrics for each game.

## Private Eye (the hardest game)
| seed | eval_return | auc | nonzero_rate |
|---|---|---|---|
| 42 | 0.0 | −174.93 | 0.0 |
| 123 | 100.0 | −10.19 | 1.0 |
| 456 | −1000.0 | −535.20 | 1.0 |
| **mean** | **−300.0** | **−240.10** | **0.667** |

## Tutankham (medium)
| seed | eval_return | auc | nonzero_rate |
|---|---|---|---|
| 42 | 0.0 | 0.0 | 0.0 |
| 123 | 109.6 | 97.10 | 1.0 |
| 456 | 0.0 | 0.0 | 0.0 |
| **mean** | **36.53** | **32.37** | **0.333** |

## Frostbite (hard exploration)
| seed | eval_return | auc | nonzero_rate |
|---|---|---|---|
| 42 | 280.0 | 246.34 | 1.0 |
| 123 | 270.0 | 244.15 | 1.0 |
| 456 | 96.0 | 85.16 | 1.0 |
| **mean** | **215.33** | **191.88** | **1.0** |

## Reading the dynamics

This is the floor I expected, and the shape of the failure is informative. Two of the three games
barely move under undirected exploration. On **Tutankham**, only seed 123 ever finds the reward
(eval 109.6); seeds 42 and 456 both flatline at exactly 0 on *every* metric — `nonzero_rate` 0.0,
`auc` 0.0 — i.e. those two seeds never saw a single reward across all of training. A mean of 36.5
that is really "one seed worked, two found nothing" is the signature of pure-luck exploration: the
outcome is decided by whether the initial random walk happens to trip a reward, not by anything the
agent learned. The same all-or-nothing pattern, more starkly, on **Private Eye**: the per-seed
`eval_return` is {0, 100, −1000}, and the −1000 is the tell — with no directed exploration the agent
wanders into the game's large *penalties* as readily as its rewards, and the mean `eval_return` is
**negative** (−300). Undirected exploration on a deceptive game is not just unhelpful, it is
actively harmful.

The one bright spot is **Frostbite**: all three seeds are nonzero (mean 215.3, auc 191.9, tight
spread 96–280). That fits — Frostbite's early reward (jumping between ice floes) is reachable enough
that the entropy-driven jitter finds it, so even the no-bonus agent gets traction. The lesson is
clean: where the first reward is within reach of random action noise, PPO is fine; where it is not
(Tutankham on two of three seeds; Private Eye on all of them), the agent is at the mercy of luck and
can be dragged *negative*. The next step has to manufacture a reward signal where the environment
gives none — an intrinsic bonus that pulls the agent toward states it has not seen, so exploration
becomes directed rather than a coin flip. The seed-to-seed variance here (a 1100-point swing on
Private Eye, 0-vs-110 on Tutankham) is itself the thing to beat: I want a method whose worst seed
isn't "found nothing."
