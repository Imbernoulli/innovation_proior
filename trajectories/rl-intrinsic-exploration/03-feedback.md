# Feedback after step 3 — RND

Real leaderboard numbers (`baseline:rnd`, `is_final,true`), seeds {42, 123, 456} and mean.

## Private Eye
| seed | eval_return | auc | nonzero_rate |
|---|---|---|---|
| 42 | 0.0 | −260.65 | 0.0 |
| 123 | 100.0 | −98.47 | 1.0 |
| 456 | 2756.0 | 106.43 | 1.0 |
| **mean** | **952.0** | **−84.23** | **0.667** |

## Tutankham
| seed | eval_return | auc | nonzero_rate |
|---|---|---|---|
| 42 | 116.2 | 104.84 | 1.0 |
| 123 | 102.2 | 51.86 | 1.0 |
| 456 | 82.6 | 48.44 | 0.8 |
| **mean** | **100.33** | **68.38** | **0.933** |

## Frostbite
| seed | eval_return | auc | nonzero_rate |
|---|---|---|---|
| 42 | 216.0 | 205.42 | 1.0 |
| 123 | 250.0 | 213.01 | 1.0 |
| 456 | 286.0 | 251.35 | 1.0 |
| **mean** | **250.67** | **223.26** | **1.0** |

## Reading the dynamics

This is the breakthrough I was watching for, and it's exactly on the game ICM couldn't touch.

**Private Eye finally cracks.** ICM was a flat 0.0 on all three seeds; RND gets a mean `eval_return`
of **952** — and the story is in the seeds: {0, 100, **2756**}. Seed 456 reached 2756, the only time
any baseline registers a substantial Private Eye return, and seed 123 also found reward (100,
`nonzero_rate` 1.0). The global, slowly-decaying novelty signal did what the local prediction-error
bonus could not: it kept marking far-flung states as unfamiliar, so the agent had a reason to keep
moving past the first mastered region and across the long reward-free gap. That is the single most
important result in the ladder so far — RND is the only baseline to get meaningful traction on the
hardest game. This is my basis for ranking RND **above** ICM: on the axis the task cares most about
(hard-exploration, sparse-reward discovery), RND is the only one that moves Private Eye, while ICM is
identically zero there.

**But it's fragile, and the failure mode is visible.** The Private Eye `auc` mean is still **negative**
(−84.2), and seed 42 is a flat 0.0 with the *worst* auc of any RND seed (−260.65) — it never found a
single reward and accumulated penalties along the way. So the 952 mean is one strong seed (456),
one modest seed (123), and one dead seed (42): `nonzero_rate` 0.667. The negative auc even on the
winning seeds (123 at −98.5) says the agent spent a lot of training in the red before late returns
pulled the final eval up. RND can reach Private Eye, but only sometimes, and not stably.

**Tutankham and Frostbite held, with a small trade.** Tutankham stayed strong (mean 100.3, close to
ICM's 109) though seed 456 slipped to `nonzero_rate` 0.8 — slightly less reliable than ICM's perfect
1.0 there. Frostbite is the interesting trade: RND's three seeds are tight and all-nonzero
{216, 250, 286}, mean 250.7 — *more dependable* than ICM, but RND has no Frostbite jackpot like ICM's
3066 seed, so its Frostbite mean (250.7) is far below ICM's variance-inflated 1156. RND traded ICM's
high-variance Frostbite peak for consistency, and bought the Private Eye breakthrough with it.

So RND is the strongest baseline: it is the only one that meaningfully explores the hardest game, and
it does so with a simpler, more stable module than ICM. Its limitation is now sharp and it is *not*
"the signal is too local" anymore — it's that the novelty is purely **lifelong/global** and decays
*monotonically across training*. Once a region's global novelty has worn off, RND gives the agent no
reason to *re-traverse* it — and Private Eye, which needs the agent to walk back through cleared
ground episode after episode to extend its reach, is exactly where that bites: two of three seeds
(42 and, partly, 123) couldn't sustain it, and the auc stayed negative. RND has no *within-episode*
notion of novelty at all: at the start of every fresh episode a state it has globally mastered looks
just as stale as it did at the end of the last one, so there is nothing pulling the agent to
re-explore. The next step has to add that missing timescale — a novelty that *resets each episode* so
the agent never gives up re-exploring — while keeping RND's global signal to down-weight what's truly
mastered.
