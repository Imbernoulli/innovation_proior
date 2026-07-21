The switch moved exactly the split I predicted and exactly *not* the one I worried about — and that
contrast is the diagnosis. Seed 42: AIME24 0.062, AMC23 0.325, MATH-500 0.250. AMC23 is the clear win, up
from the floor's 0.217, because conditioning the switch let demonstration-shaped problems get bootstrapped
while the model kept practicing the ones it could partly do. But the two numbers I was watching barely moved.
MATH-500 went `0.244 → 0.250`, flat — the lift off the floor I expected from keeping contrast prompts on RL
did not materialize at scale. And AIME24 stayed pinned at 0.062, exactly the pure-SFT value — the tell I
flagged: on hard problems the model fails all eight rollouts on nearly every prompt, so the gate fires SFT on
almost all of them and the switch collapses back toward pure SFT. In problem-counts: AMC23 `8.7 → 13.0`, a
real `+4.3`; MATH-500 `122 → 125`, `+3` out of 500, inside run-to-run wobble; AIME24 dead flat, `1.9 → 1.9`.
The entire measurable benefit landed on the 40-problem split and none reached the 500-problem split.

The reason is structural: the switch still routes every all-wrong prompt to SFT, and on this weak backbone
over a broad OpenR1 mixture that fraction is high — a prompt the model solves with probability `p` is routed
to SFT with probability `(1 − p)^8`, so over prompts with `p` in `0.05`–`0.1` that is `0.66`–`0.43` handed to
imitation every step. The switch turned "100% SFT" into "roughly half SFT," denting the imitation tax without
removing it; on MATH-500, where success depends on the model generalizing its own reasoning across 500 varied
prompts rather than reproducing teacher traces, the residual tax is enough to keep the score flat. So the
natural experiment is the opposite extreme from the floor: what does *never* imitating buy? If the imitation
tax caps MATH-500, the rule that pays it zero times — pure on-policy GRPO — isolates it.

Pure GRPO keeps every prompt's on-policy group and never switches. Why GRPO is the right on-policy learner,
and why that is the reason it should beat the switch on MATH-500: the reward is the verifier, which reads the
entire solution and emits one number effectively at the last token, with nothing in between. PPO would train
a policy-sized value network `V_ψ` to smear that terminal scalar backward into per-token credit via GAE — but
every GAE term needs an accurate `V` at every one of up to 8192 intermediate positions, regressed from a
single supervising bit at the end, which is exactly what the last-token-only reward makes hardest. A 1.5B
critic also roughly doubles the trainable footprint on the same 4×H200 box and must be optimized in lockstep;
when `V` is badly fit, `δ_t = r_t + γV(s_{t+1}) − V(s_t)` is a difference of two noisy adjacent estimates
whose noise feeds biased, high-variance advantages straight into the policy. I want PPO's stable clipped
update and a variance-reducing baseline, but not this learned per-token critic.

Can I get a baseline without learning `V`? A baseline only has to be independent of the action scored and
close to the expected return; the subtraction is free in expectation. For each prompt I already draw a group
of eight rollouts with verifier rewards `R_i`, and the empirical mean is a Monte-Carlo estimate of the
question's response-level value under the rollout policy — the very quantity `V` approximated, at the only
granularity the reward exists, for free. `R_i` is one of the terms in its own mean, so the full mean is not
strictly independent; the leave-one-out gap is `(R_i − m)/(G−1)`, a fraction of the advantage scale at `G = 8`
that the normalization swallows — so I use the full group mean. It is the *right* baseline anyway, because the
reward here is comparative, so a per-question relative baseline matches the signal.

The spread of `R_i` differs wildly across questions, so raw `R_i − mean` would let easy and hard prompts
contribute advantages of incomparable scale; normalize by the group std, `Â_i = (R_i − mean(R)) / (std(R) +
ε)`, a per-question z-score. (The `ε` floors the denominator; an all-equal group has `std = 0` and contributes
no gradient — the same degeneracy the switch caught, and here, with no switch, I accept it.) With a single
terminal reward there is no within-sequence signal, so broadcast the scalar advantage to every token,
`Â_{i,t} = Â_i` — outcome supervision, the simplest thing consistent with the information I have; inventing a
within-sequence gradient from one terminal bit is the critic's impossible regression in cheaper clothes. That
makes the per-token coefficient uniform within a response and equal to `Â_i` — a real-valued signed number,
exactly the graded reinforcement the demonstration floor lacked: the `+1.62` rollout pushed harder than a
`+0.5` one, the `−0.54` rollouts pushed *down*. Keep PPO's clipped surrogate for the update, with the
KL-to-reference moved into the loss (k3 estimator, non-negative per token). That is GRPO: PPO's machinery with
a free group-relative baseline standing in for the GAE advantage, no value network.

The edit surface is only the router — `adv_estimator=grpo` is frozen — and pure GRPO is the router that never
switches: keep every on-policy group, add nothing off-policy, `del on_solve_num; return 0, 0, 0`. The
demonstration is never touched and the imitation tax is paid exactly zero times. What the harness gives back
is the clean GRPO update on every prompt's eight rollouts; what it takes away is any bootstrap on the
all-wrong prompts — those contribute zero gradient and the model must discover those solutions on its own or
not at all. Pure on-policy means exploration bounded by the base model.

Where HPT fired SFT on every all-wrong prompt, I fire it on none. I *lose* the bootstrap on genuinely stuck
prompts — quantify where it bites hardest: the floor's AIME24 `0.062` puts the per-rollout solve probability
near `0.06`, and at `p = 0.06` a prompt fails all eight with probability `0.94^8 ≈ 0.61`, so around six in ten
AIME prompts hand GRPO an all-equal group and zero gradient every step. But I *gain* the entire MATH-500 mass:
every contrast prompt — and on the broad in-distribution split there are many, since the 1.5B math model can
partly solve a large fraction of MATH-500 — now learns from its own graded, signed rollouts instead of copying
a teacher.

The falsifiable prediction is on the broad split. If the imitation tax capped MATH-500 at 0.250, removing it
entirely should lift MATH-500 by something an order of magnitude larger than the switch's `+3` — tens of
problems out of 500, a jump of order `0.1`. If MATH-500 instead lands within a few problems of 0.250 again,
the tax was not what capped it and the diagnosis is wrong. AIME24 I expect to stay near 0.062 (pure RL has no
bootstrap and the model cannot sample correct AIME solutions); if it rises anyway, the model solves more AIME
prompts than I think and the switch was wasting them on SFT. AMC23 is the genuine risk — its switch win came
precisely from bootstrapping demonstration-shaped problems, so pure GRPO might give some back; if it drops
below 0.325, that is the cost of never imitating, and the leaderboard direction says which of the broad-split
gain and small-split loss dominates on this backbone.
