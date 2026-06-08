RND's feedback hands me a clean, specific failure to fix. RND was the breakthrough — the only baseline
to get real Private Eye return (seed 456 at 2756) — but its mean there is still negative in `auc`, two
of three seeds couldn't sustain it (seed 42 a flat zero, worst `auc` of the run), and the whole thing
is fragile. And the *reason* is not "the signal is too local" — that was ICM's problem, which RND
already fixed by going global. RND's problem is the opposite end: its novelty is purely *lifelong*. The
distillation error on a region only ever decreases across training, and it has no within-episode memory
whatsoever. So once a region's global novelty has worn off, RND gives the agent no reason to walk back
through it — and at the start of every fresh episode, a globally-mastered state looks exactly as stale
as it did at the end of the last one. There is nothing that resets.

Stare at what Private Eye demands and the gap is obvious: to extend its reach the agent has to
re-traverse cleared ground, episode after episode, because the only path to the next undiscovered area
runs through the old one. A lifelong-only bonus has, by construction, stopped paying for that
traversal. The fix is forced — I need a *second* novelty timescale that *resets every episode*, sharp
within an attempt and blind across attempts, sitting on top of RND's slow global signal so the agent
never gives up re-exploring. Here is the two-timescale construction.